#!/usr/bin/env python3
"""Test the prediction storage and visualization pipeline.

This script runs a quick test of the entire pipeline with minimal iterations
to verify everything works correctly before running full benchmarks.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add BayBE root directory to Python path
script_dir = Path(__file__).resolve().parent
baybe_root = script_dir.parent.parent.parent.parent
sys.path.insert(0, str(baybe_root))

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ Success!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False

def main():
    """Run the complete pipeline test."""
    print("🧪 Testing TL Predictions Pipeline")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Test 1: Store predictions (test mode)
    print("\n1️⃣ Testing prediction storage (test mode)...")
    cmd1 = [
        sys.executable, "store_tl_predictions.py", 
        "--test",
        "--benchmark", "forrester_noise_05"
        # Note: output_dir defaults to script directory now
    ]
    
    success1 = run_command(cmd1, "Running prediction storage")
    
    if not success1:
        print("❌ Pipeline test failed at prediction storage step")
        return False
    
    # Check if JSON file was created (in script directory)
    json_file = "forrester_noise_05_predictions.json"
    if os.path.exists(json_file):
        print(f"   ✅ JSON file created: {json_file}")
        
        # Check file size
        file_size = os.path.getsize(json_file)
        print(f"   📊 File size: {file_size:,} bytes")
        
        if file_size < 100:
            print("   ⚠️  Warning: File seems very small")
    else:
        print(f"   ❌ JSON file not found: {json_file}")
        return False
    
    # Test 2: Visualize predictions
    print("\n2️⃣ Testing visualization...")
    cmd2 = [
        sys.executable, "visualize_predictions.py",
        "--benchmark", "forrester_noise_05"
        # Note: input_dir and output_dir default to script directory now
    ]
    
    success2 = run_command(cmd2, "Running visualization")
    
    if not success2:
        print("❌ Pipeline test failed at visualization step")
        return False
    
    # Check if plots were created
    plot_dirs = [
        "predictions/forrester_noise_05/mc_iterations",
        "predictions/forrester_noise_05/models"
    ]
    
    plots_created = 0
    for plot_dir in plot_dirs:
        if os.path.exists(plot_dir):
            png_files = list(Path(plot_dir).glob("*.png"))
            plots_created += len(png_files)
            print(f"   ✅ Created {len(png_files)} plots in {plot_dir}")
        else:
            print(f"   ❌ Plot directory not found: {plot_dir}")
    
    print(f"\n📈 Total plots created: {plots_created}")
    
    if plots_created > 0:
        print("✅ Pipeline test PASSED!")
        print("\n🚀 Ready to run full benchmarks with:")
        print("   python store_tl_predictions.py --n_mc_iterations 5 --n_doe_iterations 10")
        print("   python visualize_predictions.py")
        return True
    else:
        print("❌ Pipeline test FAILED - no plots created")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)