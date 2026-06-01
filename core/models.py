import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CNN_LSTM(nn.Module):
    def __init__(self, num_channels=21, num_classes=2, lstm_hidden=128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.spatial_dropout = nn.Dropout2d(0.3)
        
        self.lstm = nn.LSTM(
            input_size=128 * 5,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.4,
            batch_first=True,
            bidirectional=True
        )
        
        self.temporal_attention = TemporalAttentionPool(lstm_hidden * 2)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))
        x = self.spatial_dropout(x)
        
        B, C, F_dim, T_dim = x.size()
        x = x.view(B, C * F_dim, T_dim)
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
