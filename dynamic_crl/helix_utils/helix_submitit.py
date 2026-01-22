# dynamic_crl/helix_utils/helix_submitit.py
from __future__ import annotations
import os, shlex, submitit, subprocess
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
import re, time

# --------- internals ---------
def _default_run_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-")

def _ws_path(ws_name: str) -> Path:
    out = subprocess.check_output(f"ws_find {shlex.quote(ws_name)}", shell=True, text=True).strip()
    if not out:
        raise RuntimeError(f"Workspace '{ws_name}' not found. Run `ws_list -a`.")
    return Path(out)

def _repo_root_from_this_module() -> Path:
    # .../dynamic_crl/helix_utils/helix_submitit.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]

def _configure_executor(
    ex: submitit.AutoExecutor,
    *,
    cpu: bool,
    minutes: int,
    cpus: int,
    mem_gb: int,
    gpu_type: Optional[str],
    part_cpu: str,
    part_gpu: str,
    chdir: Path,
):
    base = dict(timeout_min=minutes, nodes=1, tasks_per_node=1, cpus_per_task=max(1, cpus), mem_gb=mem_gb)
    addl = {"export": "ALL", "chdir": str(chdir)}
    if cpu:
        ex.update_parameters(slurm_partition=part_cpu, slurm_additional_parameters=addl, **base)
    else:
        gres = f"gpu:{gpu_type}:1" if gpu_type else "gpu:1"
        addl_gpu = {**addl, "gres": gres}
        ex.update_parameters(slurm_partition=part_gpu, slurm_additional_parameters=addl_gpu, **base)

# ----------------------
# Save root resolution
# ----------------------
def resolve_run_root(current_dir) -> Path:
    """
    Cluster runs (Submitit/SLURM):
      - Prefer RESULTS_ROOT / HELIX_RUN_ROOT if we're inside a scheduled job.
    Local runs (no Submitit/SLURM):
      - Save next to this script under ./runs
    Also supports EXPL_CFG pointing at a pinned config (kept for compatibility).
    """
    is_cluster = bool(
        os.environ.get("SUBMITIT_FOLDER") or
        os.environ.get("SLURM_JOB_ID") or
        os.environ.get("SLURM_JOB_NAME")
    )

    if is_cluster:
        for k in ("RESULTS_ROOT", "HELIX_RUN_ROOT"):
            v = os.environ.get(k, "").strip()
            if v:
                return Path(v).resolve()

        # Fallback: EXPL_CFG might be inside a run dir
        cfg_env = os.environ.get("EXPL_CFG", "").strip()
        if cfg_env:
            p = Path(cfg_env).resolve()
            if p.parent.name == "_pinned_cfg":
                return p.parent.parent
            return p.parent

    # Local default: save next to code
    return (Path(current_dir) / "runs").resolve()

# --------- public API ---------
def run_with_submitit(
    main_fn: Callable[[Path], object],   # your exp_main(results_root: Path)
    *,
    submit: Optional[bool] = None,       # None -> decide via HELIX_SUBMIT env; True -> submit; False -> run local
    ws_name: str = "crl-exp",
    cpu: bool = True,
    minutes: int = 30,
    cpus: int = 4,
    mem_gb: int = 8,
    gpu_type: Optional[str] = None,      # e.g. "A40"/"A100" when cpu=False
    part_cpu: str = "cpu-single",
    part_gpu: str = "gpu-single",
    link_target_subdir: str = "runs",    # subfolder inside workspace for results
    run_name: Optional[str] = None,
) -> None:
    """
    Usage in experiments:

        def exp_main(results_root: Path):
            save_dir = results_root / "my_experiment"
            ...

        if __name__ == "__main__":
            run_with_submitit(exp_main)

    Local:  results_root = <repo>/dynamic_crl/experiments_results/<run_name>
    Helix:  results_root = <WS>/<link_target_subdir>/<run_name>
    """
    repo_root = _repo_root_from_this_module()

    # Decide local vs submit
    if submit is None:
        submit = os.environ.get("HELIX_SUBMIT", "").strip() not in ("", "0", "false", "False")

    # Ensure repo importable at submit & unpickle time
    os.environ["PYTHONPATH"] = str(repo_root) + ":" + os.environ.get("PYTHONPATH", "")

    run_name = _slugify(run_name) if run_name else _default_run_name()

    # ----- Local run -----
    if not submit:
        results_root = repo_root / "dynamic_crl"
        results_root.mkdir(parents=True, exist_ok=True)
        os.chdir(str(repo_root))
        start = time.time()
        out = main_fn(results_root)
        dur = time.time() - start
        print(f"[local] Finished in {dur/60:.2f} minutes  results_root={results_root}")
        return out

    # ----- Helix run -----
    ws_dir = _ws_path(ws_name)
    logs = ws_dir / "submitit_logs" / run_name
    logs.mkdir(parents=True, exist_ok=True)

    results_root = ws_dir / link_target_subdir
    results_root.mkdir(parents=True, exist_ok=True)

    def _job_wrapper():
        import sys
        sys.path.insert(0, str(repo_root))
        os.environ["PYTHONPATH"] = f"{repo_root}:{os.environ.get('PYTHONPATH','')}"
        os.chdir(str(repo_root))  # stable CWD on node

        # (optional) make multiprocessing robust
        try:
            import multiprocessing as mp
            mp.set_start_method("fork", force=True)
        except Exception:
            pass

        # caches and knobs
        os.environ.setdefault("PIP_CACHE_DIR", str(ws_dir / "cache" / "pip"))
        os.makedirs(os.environ["PIP_CACHE_DIR"], exist_ok=True)
        os.environ.setdefault("WS_DIR", str(ws_dir))
        os.environ.setdefault("RESULTS_ROOT", str(results_root))
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(cpus, 8))))
        os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
        os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

        print(f"[submitit] CWD={os.getcwd()}")
        print(f"[submitit] results_root={results_root}")

        start = time.time()
        out = main_fn(results_root)
        dur = time.time() - start
        print(f"[submitit] Job finished in {dur/60:.2f} minutes")
        return out

    ex = submitit.AutoExecutor(folder=str(logs))
    _configure_executor(
        ex, cpu=cpu, minutes=minutes, cpus=cpus, mem_gb=mem_gb,
        gpu_type=gpu_type, part_cpu=part_cpu, part_gpu=part_gpu, chdir=repo_root
    )

    os.environ["SUBMITIT_EXECUTOR"] = "helix"
    job = ex.submit(_job_wrapper)
    print(f"[submitit] submitted {job.job_id}  logs: {logs}  results_root={results_root}")
