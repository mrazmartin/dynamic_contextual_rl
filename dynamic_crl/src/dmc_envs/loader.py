# dynamic_crl/dc_envs/loader.py
from __future__ import annotations
import os, sys, inspect
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import dm_env
from dm_control import suite as dmc_suite
import mujoco

# Ensure project root on path (adjust levels if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../..')
sys.path.append(project_root)

# Import custom domains that expose SUITE and/or top-level factories
from dynamic_crl.src.dmc_envs.dc_envs.dmc_tasks import dmc_go2  # registers SUITE on import
from dynamic_crl.src.dmc_envs.dc_envs.dmc_tasks import quadruped

try:
    from carl.utils.types import Context  # type: ignore
except Exception:
    Context = Dict[str, Any]  # fallback typing


# ---- Registry of custom domains (string -> module) ----
_CUSTOM_DOMAINS: Dict[str, Any] = {
    "go2": dmc_go2,
    "quadruped": quadruped,
}

def _task_map_from_suite(module) -> Dict[str, Callable]:
    """Turn containers.TaggedTasks() into {task_name: factory_fn}."""
    try:
        return {name: fn for name, fn in module.SUITE}
    except Exception:
        return {}

def compile_from_file(xml_path: str | Path):
    """Compile a MuJoCo model directly from an XML path and return (model, data)."""
    xml_path = Path(xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def load_dmc_env(
    domain_name: str,
    task_name: str,
    context: Optional[Context] = None,
    task_kwargs: Optional[Dict[str, Any]] = None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    visualize_reward: bool = False,
) -> dm_env.Environment:
    """
    CARL-like loader: try our custom domains first, then fall back to dm_control.suite.load.

    Custom domain:
      - Prefer a top-level factory named `task_name`, e.g., dmc_go2.stand_up(...)
      - Else, look up in module.SUITE (iterable of (name, fn))
      - Call with (context=..., environment_kwargs=..., **task_kwargs) when supported

    Stock dm_control:
      - Use dm_control.suite.load(domain_name, task_name, ...)
      - (No `context` there — pass via task_kwargs yourself if needed)
    """
    ctx = context or {}
    tk  = dict(task_kwargs or {})
    ek  = dict(environment_kwargs or {})

    # ---------- 1) Try custom domain ----------
    module = _CUSTOM_DOMAINS.get(domain_name)
    if module is not None:
        # (a) top-level factory def XXXX(...)
        factory = getattr(module, task_name, None)

        # (b) else from SUITE tag registry
        if factory is None:
            factory = _task_map_from_suite(module).get(task_name)

        if factory is None:
            raise ValueError(
                f"Task '{task_name}' not found in custom domain '{domain_name}'. "
                f"Available: {list(_task_map_from_suite(module).keys())}"
            )

        # Try CARL-style signature first
        try:
            env = factory(context=ctx, environment_kwargs=ek, **tk)
        except TypeError:
            # Fallback for factories without context/environment_kwargs
            raise ValueError(
                f"Issue calling factory for {factory.__name__} because the arguments do not match:\ntk: {tk}\nek: {ek}"
            )
            env = factory(**tk)

        # Try toggling visualize_reward if the task exposes it
        try:
            setattr(env.task, "visualize_reward", visualize_reward)
        except Exception:
            pass

        return env

    # ---------- 2) Fallback: stock dm_control ----------
    return dmc_suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs=tk if tk else None,
        environment_kwargs=ek if ek else None,
        visualize_reward=visualize_reward,
    )
