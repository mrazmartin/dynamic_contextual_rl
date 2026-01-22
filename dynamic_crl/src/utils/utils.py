import yaml
from pathlib import Path
from typing import Any, Dict, Sequence, Callable, List, Optional, Tuple
import numpy as np
import os

def load_cfg(yaml_path: Path) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

# ------------------------
# naming experiment runs
# ------------------------
import re

def sanitize(x):
    s = str(x).replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]+", "", s)

def condition_name(ctx_key, dyn_kind, dyn_params, min_v, max_v, train_ctxs_kind, single_value):
    parts = [ctx_key, f"dyn_{dyn_kind}"]
    for k, v in sorted(dyn_params.items()):
        parts.append(f"{k[:6]}{sanitize(v)}")
    parts.append(f"min{sanitize(min_v)}_max{sanitize(max_v)}")
    parts.append(f"train_{'yaml' if train_ctxs_kind=='yaml' else 'single'}")
    if train_ctxs_kind != "yaml":
        parts.append(f"V{sanitize(single_value)}")
    return "__".join(parts)

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