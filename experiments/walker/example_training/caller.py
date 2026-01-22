import os
import sys
import json
import subprocess
from pathlib import Path

# --- Configuration ---
# 1. Where are the files?
CURRENT_DIR = Path(__file__).parent.resolve()
TRAIN_SCRIPT = CURRENT_DIR / "dc_walker_training.py"
CFG_PATH = CURRENT_DIR / "walker_config.yaml"

# 2. Define the Test Condition (Simulating a submitter job)
# We test: Sinusoidal dynamics on the "triple" pool ([-0.6, 0.0, 0.6])
TEST_PAYLOAD = {
    "ctx_key": "COM_X",
    "dyn_kind": "sinusoidal",  # Matches the key in dyn_factory_from_condition
    "dyn_params": {
        "amplitude": 0.3,
        "period": 200
    },
    "dyn_seed": 42,
    "train_ctxs_kind": "pool",
    "train_pool": "triple",     # Must exist in walker_config.yaml train_pools
    "obs_mode": "none",         # "live" = observe context or 'none'
    "single_value": None
}

# 3. Environment Variables for the Runner
env = os.environ.copy()
env.update({
    # Paths
    "CFG_PATH": str(CFG_PATH),
    "RESULTS_ROOT": str(CURRENT_DIR / "runs"), # output locally
    
    # Condition Data
    "COND_NAME": "local_test_sinusoidal",
    "COND_PAYLOAD_JSON": json.dumps(TEST_PAYLOAD),
    
    # Run Settings
    "SEED": "100",           # Test seed
    "EVAL_N_ENVS": "2",      # Keep low for local test
    "EVAL_BASE_SEED": "999",
    "NORM_MODE": "ctx",      # Context normalization
    
    # Hardware/Backend settings (Force CPU/Headless for safety)
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "", # Force CPU
    "SDL_VIDEODRIVER": "dummy"
})

def run_test():
    print("="*60)
    print(f"Starting Local Walker Test")
    print(f"Script: {TRAIN_SCRIPT}")
    print(f"Config: {CFG_PATH}")
    print(f"Payload: {json.dumps(TEST_PAYLOAD, indent=2)}")
    print("="*60)

    if not TRAIN_SCRIPT.exists():
        print(f"Error: Could not find {TRAIN_SCRIPT}")
        return

    if not CFG_PATH.exists():
        print(f"Error: Could not find {CFG_PATH}")
        return

    try:
        # Run the experiment script as a subprocess
        # This ensures it runs exactly as it would via the submitter
        subprocess.check_call([sys.executable, str(TRAIN_SCRIPT)], env=env)
        print("\n" + "="*60)
        print("SUCCESS: Test run finished without errors.")
        print(f"Check results in: {CURRENT_DIR / 'runs'}")
        print("="*60)
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*60)
        print(f"FAILURE: Process crashed with exit code {e.returncode}")
        print("="*60)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

if __name__ == "__main__":
    run_test()