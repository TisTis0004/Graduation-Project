import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock2d(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)
        return pooled

class Spectrogram_CNN_LSTM(nn.Module):
    """
    Upgraded CNN-LSTM with:
    1. InstanceNorm2d (domain-invariant, replaces BatchNorm2d)
    2. Temporal Attention Pooling (replaces last-timestep)
    3. Multi-scale convolutions (parallel 3x3 + 5x5 kernels)
    """
    def __init__(self, num_channels=18, num_classes=2, lstm_hidden=64):
        super().__init__()

        # Channel attention (Squeeze-and-Excite)
        self.se = SEBlock2d(num_channels, reduction=2)

        # === MULTI-SCALE CNN with InstanceNorm ===
        # Block 1: 18 -> 32
        self.conv1_3x3 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv1_5x5 = nn.Conv2d(num_channels, 16, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Block 2: 32 -> 64
        self.conv2_3x3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2_5x5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Block 3: 64 -> 128
        self.conv3_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_5x5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.spatial_dropout = nn.Dropout2d(p=0.3)

        # Freq dimension reduces from 40 -> 20 -> 10 -> 5
        cnn_out_channels = 128
        freq_out = 5
        lstm_input_size = cnn_out_channels * freq_out

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.4,
            batch_first=True,
            bidirectional=True
        )

        # === ATTENTION TEMPORAL POOLING ===
        self.temporal_attention = TemporalAttentionPool(lstm_hidden * 2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def _cnn_block(self, x, conv_3x3, conv_5x5, norm, pool):
        """Multi-scale convolution: concatenate 3x3 and 5x5 features."""
        out_3 = conv_3x3(x)
        out_5 = conv_5x5(x)
        out = torch.cat([out_3, out_5], dim=1)
        out = norm(out)
        out = F.relu(out)
        out = pool(out)
        return out

    def forward(self, x):
        # x: [B, C, F, T]
        x = self.se(x)

        x = self._cnn_block(x, self.conv1_3x3, self.conv1_5x5, self.norm1, self.pool1)
        x = self._cnn_block(x, self.conv2_3x3, self.conv2_5x5, self.norm2, self.pool2)
        x = self._cnn_block(x, self.conv3_3x3, self.conv3_5x5, self.norm3, self.pool3)
        x = self.spatial_dropout(x)

        B, C, F, T = x.size()
        x = x.view(B, C * F, T)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        attended = self.temporal_attention(lstm_out)

        out = self.dropout(attended)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

class EEGNet(nn.Module):
    def __init__(self, n_chans=41, n_classes=2, fs=250, window_sec=10):
        super().__init__()
        F1, D = 8, 2
        F2 = F1 * D
        kernel_1 = 64
        
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_1), padding="same", bias=False)
        self.bnorm_temporal = nn.BatchNorm2d(F1)
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bnorm_1 = nn.BatchNorm2d(F1 * D)
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.drop_1 = nn.Dropout(0.25)
        
        self.conv_separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False)
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bnorm_2 = nn.BatchNorm2d(F2)
        self.pool_2 = nn.AvgPool2d((1, 8))
        self.drop_2 = nn.Dropout(0.25)
        
        T_out = (fs * window_sec) // 32
        self.final_layer = nn.Sequential(
            nn.Conv2d(F2, n_classes, (1, T_out), bias=True),
            nn.Flatten()
        )

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = nn.functional.elu(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = nn.functional.elu(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        return self.final_layer(x)
