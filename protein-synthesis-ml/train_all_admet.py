"""
train_all_admet.py

Train all ADMET models in sequence.
This script trains all 11 ADMET tasks automatically.
"""

import subprocess
import sys
import os

# Check if we're using the right Python
python_path = sys.executable
if "anaconda" not in python_path and "conda" not in python_path:
    print("⚠️  WARNING: Not using conda Python!")
    print(f"   Current Python: {python_path}")
    print("   Please run: conda activate base")
    print("   Then use: python train_all_admet.py")
    print()
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

try:
    from admet_loader import ADMET_TASKS
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure you're using conda Python and have installed dependencies:")
    print("  conda activate base")
    print("  python -m pip install pytdc")
    sys.exit(1)


def train_all_admet_models():
    """Train all ADMET models sequentially."""
    print("=" * 70)
    print("Training All ADMET Models")
    print("=" * 70)
    print(f"\nWill train {len(ADMET_TASKS)} ADMET tasks:\n")
    
    for task_key in ADMET_TASKS.keys():
        task_config = ADMET_TASKS[task_key]
        print(f"  - {task_key} ({task_config['type']})")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    failed = []
    succeeded = []
    
    for idx, task_key in enumerate(ADMET_TASKS.keys(), 1):
        print(f"\n[{idx}/{len(ADMET_TASKS)}] Training {task_key}...")
        print("-" * 70)
        
        try:
            result = subprocess.run(
                [sys.executable, "train_admet.py", "--task", task_key],
                check=True,
                capture_output=False
            )
            succeeded.append(task_key)
            print(f"✅ {task_key} training complete")
        except subprocess.CalledProcessError as e:
            failed.append(task_key)
            print(f"❌ {task_key} training failed")
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            print(f"Completed: {len(succeeded)}/{len(ADMET_TASKS)}")
            print(f"Failed: {len(failed)}")
            if failed:
                print(f"Failed tasks: {', '.join(failed)}")
            sys.exit(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"✅ Succeeded: {len(succeeded)}/{len(ADMET_TASKS)}")
    if succeeded:
        print(f"   {', '.join(succeeded)}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print(f"   {', '.join(failed)}")
        print("\nYou can retry failed tasks individually:")
        for task in failed:
            print(f"   python train_admet.py --task {task}")
    else:
        print("\n🎉 All ADMET models trained successfully!")
        print("You can now use the full screening pipeline in the Streamlit UI.")


if __name__ == "__main__":
    train_all_admet_models()

