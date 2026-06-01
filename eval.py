import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Interface for EEG Seizure Detection")
    parser.add_argument("--model", type=str, required=True, choices=["eegnet", "cnn_lstm"], help="Which model architecture to evaluate")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file (.pt)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the evaluation dataset manifest (.jsonl)")
    parser.add_argument("--n_chans", type=int, default=21, help="Number of channels")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    
    args, unknown = parser.parse_known_args()

    print("=" * 60)
    print(f"Starting evaluation for model: {args.model.upper()}")
    print("=" * 60)

    if args.model == "eegnet":
        try:
            cmd = [
                sys.executable, "core/eval_single_model.py",
                "--ckpt", args.ckpt,
                "--manifest", args.manifest,
                "--n_chans", str(args.n_chans),
                "--n_classes", str(args.n_classes)
            ] + unknown
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Evaluation failed with exit code {e.returncode}")
            sys.exit(e.returncode)
    else:
        print(f"Error: Model {args.model} is not fully implemented yet.")
        sys.exit(1)

if __name__ == "__main__":
    main()
