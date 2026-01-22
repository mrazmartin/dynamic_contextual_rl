import yaml
from pathlib import Path
from typing import Any, Dict, Sequence, Callable, List, Optional, Tuple
import numpy as np
import os

def load_cfg(yaml_path: Path) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
    
# ----------------------
# Utilities / Seeding
# ----------------------
def seed_everything(seed: int) -> None:
    """
    Stronger seeding for reproducibility (esp. for eval):
    - Python/NumPy/Torch RNGs
    - Single-threaded BLAS to avoid nondeterministic scheduling
    - cuDNN deterministic off & benchmark off (if present)
    """
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
        # Keep deterministic algorithms relaxed to avoid forbidden ops in SB3/Gym
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    except Exception:
        pass