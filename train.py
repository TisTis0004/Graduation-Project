import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Unified Training Interface for EEG Seizure Detection")
    parser.add_argument("--model", type=str, required=True, choices=["eegnet"], help="Which model to train")
    
    # Allow passing extra arguments to the underlying script
    args, unknown = parser.parse_known_args()

    print("=" * 60)
    print(f"Starting training for model: {args.model.upper()}")
    print("=" * 60)

    if args.model == "eegnet":
        # Call the eegnet training script directly
        try:
            subprocess.run([sys.executable, "core/train_eegnet.py"] + unknown, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Training failed with exit code {e.returncode}")
            sys.exit(e.returncode)
    else:
        print(f"Error: Model {args.model} is not fully implemented yet.")
        sys.exit(1)

if __name__ == "__main__":
    main()
