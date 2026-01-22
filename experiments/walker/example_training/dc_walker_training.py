#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, csv, time, json
from pathlib import Path
from typing import Any, Dict, Sequence, Callable, List, Tuple

import numpy as np
import torch
import yaml
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

# ========= Project paths =========
current_dir = Path(__file__).parent.resolve()
project_root = Path(os.path.abspath(os.path.join(current_dir, '../../../')))
sys.path.append(str(project_root))

# ========= Your modules (walker) =========
from dynamic_crl.src.utils.gym_utils import walker_env_factory
from dynamic_crl.src.dynamic_carl.env_wrappers import GymDynamicContextCarlWrapper
from carl.context.context_space import UniformFloatContextFeature
from dynamic_crl.src.dynamic_carl.base_classes import PartialContextWrapper

# same helpers you referenced in your replay script
from dynamic_crl.src.utils.gym_utils import (_get_hull_com_x, _set_hull_com_x,)

# =============================================================================
# CHANGED: Import from the Universal Scheduler Library
# =============================================================================
from dynamic_crl.src.dynamic_carl.gym_context_updates import (
    make_identity,
    make_random_walk,
    make_piecewise_constant,
    make_continuous_incrementer,
    make_sinusoidal,
    make_cosine_annealing,
    make_sudden_jump,
    Actuator  # Import base class for type hinting/inheritance
)

# =============================================================================
# NEW: Walker Actuator Definition
# =============================================================================
class WalkerComXActuator(Actuator):
    """
    Custom actuator to set the Center of Mass X position for BipedalWalker.
    """
    def __init__(self, com_mass, com_radius, com_y):
        self.mass = float(com_mass)
        self.radius = float(com_radius)
        self.y = float(com_y)

    @property
    def name(self):
        return "COM_X"

    def get(self, env):
        return _get_hull_com_x(env)

    def set(self, env, val):
        _set_hull_com_x(
            env, 
            x=float(val), 
            mass=self.mass, 
            radius=self.radius, 
            y_fixed=self.y
        )
        return float(val)

# =========================
# YAML + ENV loader (macro-driven)
# =========================
_VALID_NORM_MODES = {"none", "vec", "ctx"}

def get_train_range(cfg: Dict[str, Any], key: str) -> Tuple[float, float]:
    try:
        lo, hi = cfg["train_ranges"][key]
        lo, hi = float(lo), float(hi)
        if not (lo < hi):
            raise ValueError
        return lo, hi
    except Exception:
        raise KeyError(f"cfg.train_ranges.{key} must exist and be [lo, hi] with lo < hi.")

def choose_norm_mode(cfg: Dict[str, Any] | None) -> str:
    env_mode = os.environ.get("NORM_MODE", "").strip().lower()
    if env_mode in _VALID_NORM_MODES:
        mode = env_mode
    else:
        mode = (cfg or {}).get("norm_mode", "ctx")
        mode = str(mode).strip().lower()
        if mode not in _VALID_NORM_MODES:
            mode = "ctx"
    print(f"[cfg] norm_mode={mode}")
    return mode

def load_from_macro_env_and_yaml() -> Dict[str, Any]:
    CFG_PATH = os.environ.get("CFG_PATH") or str(current_dir / "walker_config.yaml")
    cfg = yaml.safe_load(open(CFG_PATH, "r"))
    norm_mode = choose_norm_mode(cfg)

    # --- Common env ---
    seed           = int(os.environ.get("SEED", "0"))
    eval_n         = int(os.environ.get("EVAL_N_ENVS", "4"))
    eval_base_seed = int(os.environ.get("EVAL_BASE_SEED", "7777"))

    # Try JSON payload first (preferred, from submitter)
    payload_json = os.environ.get("COND_PAYLOAD_JSON", "").strip()
    if payload_json:
        cp = json.loads(payload_json)
        # expected keys from your submitter:
        #   ctx_key, dyn_kind, dyn_params, dyn_seed, train_ctxs_kind, single_value, train_pool, obs_mode
        ctx_key   = cp["ctx_key"]
        cond_spec = {
            "dyn_kind":         cp["dyn_kind"],
            "dyn_params":       dict(cp.get("dyn_params", {})),
            "dyn_seed":         int(cp.get("dyn_seed", 0)),
            "train_ctxs_kind":  str(cp.get("train_ctxs_kind", "single")).lower(),
            "single_value":     cp.get("single_value", (cfg.get("single_defaults", {}) or {}).get(ctx_key, None)),
            "train_pool":       cp.get("train_pool", None),
            "obs_mode":         str(cp.get("obs_mode", "live")).lower(),
        }
        cond_name = os.environ.get("COND_NAME", "") or "explicit_cond"

    else:
        # ---- Fallback: legacy YAML expansion path (no JSON available) ----
        # ctx_key: prefer ENV, else first from YAML, else "COM_X"
        ctx_keys_yaml = list(cfg.get("ctx_keys", []))
        ctx_key = os.environ.get("CTX_KEY") or (ctx_keys_yaml[0] if ctx_keys_yaml else "COM_X")
        cond     = os.environ.get("COND_NAME", "").strip()

        # find the matching condition from YAML dynamic_search
        from dynamic_crl.src.utils.utils import condition_name

        def _grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
            if not grid: return [dict()]
            keys = list(grid.keys()); vals = [grid[k] for k in keys]
            import itertools
            return [{k: v for k, v in zip(keys, combo)} for combo in itertools.product(*vals)]

        train_lo, train_hi = get_train_range(cfg, ctx_key)
        if "train_contexts" in cfg and ctx_key in cfg["train_contexts"]:
            tlist = cfg["train_contexts"][ctx_key]
            min_v = float(min(tlist)); max_v = float(max(tlist))
        else:
            min_v, max_v = train_lo, train_hi
        single_default = (cfg.get("single_defaults", {}) or {}).get(ctx_key, None)

        found = None
        for spec in cfg.get("dynamic_search", []):
            dyn_kind = spec["dyn_kind"]
            tkind    = str(spec.get("train_ctxs_kind", "single")).lower()
            fixed    = dict(spec.get("fixed", {}))
            grid     = dict(spec.get("grid", {}))
            for combo in _grid(grid):
                dyn_params = {**fixed, **combo}
                cname = condition_name(
                    ctx_key=ctx_key,
                    dyn_kind=dyn_kind,
                    dyn_params={**dyn_params, "norm": norm_mode},
                    min_v=min_v, max_v=max_v,
                    train_ctxs_kind=tkind,
                    single_value=spec.get("single_value", single_default),
                )
                if cname == cond:
                    found = dict(
                        dyn_kind=dyn_kind,
                        dyn_params=dyn_params,
                        dyn_seed=int(spec.get("dyn_seed", 0)),
                        train_ctxs_kind=tkind,
                        single_value=spec.get("single_value", single_default),
                        train_pool=None,
                        obs_mode="live",
                    )
                    cond = cname
                    break
            if found: break
        if not found:
            raise KeyError(f"COND_NAME='{cond}' not found via YAML dynamic_search expansion")
        cond_spec = found
        cond_name = cond

    # --- Ranges ---
    train_lo, train_hi = get_train_range(cfg, ctx_key)

    # union over eval pools to get eval feature range (true OOD allowed in eval)
    global_min, global_max = train_lo, train_hi
    ep = cfg.get("eval_pools", {})
    for tag in ("ood_low", "ood_high", "id"):
        for v in ep.get(tag, {}).get(ctx_key, []):
            global_min = min(global_min, float(v))
            global_max = max(global_max, float(v))
    eval_feature_bounds = (global_min, global_max)

    # context normalization envelope (by union)
    context_norm_ranges = {ctx_key: (global_min, global_max)} if norm_mode == "ctx" else None

    # Resolve training values for this condition
    pools = cfg.get("train_pools", {})
    pool_name_env = os.environ.get("TRAIN_POOL", "").strip()
    pool_name_cfg = cfg.get("default_train_pool") or (list(pools.keys())[0] if pools else None)

    tkind    = str(cond_spec.get("train_ctxs_kind", "single")).lower()
    dyn_kind = cond_spec["dyn_kind"]
    obs_mode = str(cond_spec.get("obs_mode", "live")).lower()

    if tkind == "single":
        # interpret single value "min"/"max"/float/None->0.0
        sv = cond_spec.get("single_value", (cfg.get("single_defaults", {}) or {}).get(ctx_key, None))
        if isinstance(sv, str):
            sv_l = sv.lower()
            if sv_l == "min": val = float(train_lo)
            elif sv_l == "max": val = float(train_hi)
            else: val = float(sv)
        elif sv is None:
            val = 0.0
        else:
            val = float(sv)
        train_vals = [val]
        pool_name   = "baseline"
    else:
        pool_name = cond_spec.get("train_pool") or (pool_name_env if pool_name_env in pools else pool_name_cfg)
        if not pool_name:
            raise RuntimeError("No training pools found; set TRAIN_POOL or add default_train_pool in YAML.")
        train_vals = list(map(float, pools[pool_name][ctx_key]))

    # Policy for observation mode (enforced):
    # single+identity -> none; single+dynamic -> live; pool -> use given obs_mode
    is_identity = dyn_kind in ("identity", "none", "static")
    if tkind == "single":
        obs_mode = "none" if is_identity else "live"

    # eval pools
    eval_pools = cfg.get("eval_pools", {})
    eval_low   = list(map(float, eval_pools.get("ood_low",  {}).get(ctx_key, [])))
    eval_high  = list(map(float, eval_pools.get("ood_high", {}).get(ctx_key, [])))
    eval_id    = list(map(float, eval_pools.get("id",       {}).get(ctx_key, [])))

    return dict(
        cfg=cfg,
        cfg_path=CFG_PATH,
        ctx_key=ctx_key,
        cond_name=cond_name,
        cond_spec={k: v for k, v in cond_spec.items() if k in ("dyn_kind","dyn_params","dyn_seed","train_ctxs_kind","single_value")},
        seed=seed,
        eval_n_envs=eval_n,
        eval_base_seed=eval_base_seed,
        train_bounds=(train_lo, train_hi),
        eval_feature_bounds=eval_feature_bounds,
        train_vals=train_vals,
        eval_vals=dict(id=eval_id, ood_low=eval_low, ood_high=eval_high),
        context_norm_ranges=context_norm_ranges,
        pool_name=pool_name,
        obs_mode=obs_mode,
    )


# --- naming helpers ---
def _shorten_cond_name(s: str) -> str:
    # purely cosmetic shortening for TensorBoard readability
    s = s.replace("dyn_cosine_annealing", "dyn_ca")
    s = s.replace("cosine_annealing", "ca")
    s = s.replace("sinusoidal", "sin")
    s = s.replace("train_single", "single")
    s = s.replace("train_pool", "pool")
    return s

def _exp_path(results_root: Path, ctx_key: str, cond_name: str, tkind: str, pool_name: str | None, seed: int) -> tuple[Path, Path, Path, Path]:
    # Folder layout:
    # runs/<ctx>/<cond>/seed_<seed>                                          (single)
    # runs/<ctx>/<cond>__pool_<pool>/seed_<seed>                             (pool)
    parts = [ctx_key, _shorten_cond_name(cond_name)]
    if tkind != "single" and pool_name:
        parts[-1] = f"{parts[-1]}__pool_{pool_name}"
    run_dir   = results_root / "__".join(parts) / f"seed_{seed}"
    tb_dir    = run_dir / "tb"
    ckpt_dir  = run_dir / "checkpoints"
    models_dir= run_dir / "models"
    return run_dir, tb_dir, ckpt_dir, models_dir


# ===== context normalization wrapper =====
class ContextOnlyNormalize(gym.ObservationWrapper):
    def __init__(self, env, ranges: dict[str, tuple[float, float]]):
        super().__init__(env)
        self.ranges = {k: (float(lo), float(hi)) for k, (lo, hi) in ranges.items()}
        obs_space = self.env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict) or "context" not in obs_space.spaces:
            raise ValueError("ContextOnlyNormalize expects a Dict obs space with a 'context' key.")
        ctx_space = obs_space.spaces["context"]
        self._is_dict_ctx = isinstance(ctx_space, gym.spaces.Dict)

        if self._is_dict_ctx:
            all_ctx_keys = list(ctx_space.spaces.keys())
            order_from_env = getattr(self.env, "ctx_to_observe", None)
            self._ctx_keys = list(order_from_env) if (order_from_env and all(k in all_ctx_keys for k in order_from_env)) else all_ctx_keys
            self._mid_per_key, self._half_per_key, new_ctx_spaces = {}, {}, {}
            for k in self._ctx_keys:
                sub: gym.spaces.Box = ctx_space.spaces[k]
                if not isinstance(sub, gym.spaces.Box): raise ValueError(f"context subspace '{k}' must be Box")
                if sub.shape not in [(), (1,)]: raise ValueError(f"context subspace '{k}' must be scalar Box")
                lo, hi = self.ranges.get(k, (float(sub.low.flatten()[0]), float(sub.high.flatten()[0])))
                mid, half = 0.5*(lo+hi), max(1e-8, 0.5*(hi-lo))
                self._mid_per_key[k], self._half_per_key[k] = float(mid), float(half)
                new_ctx_spaces[k] = gym.spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                                   high=np.array([+1.0], dtype=np.float32),
                                                   dtype=np.float32)
            self.observation_space = gym.spaces.Dict({**{k: v for k, v in obs_space.spaces.items() if k != "context"},
                                                      "context": gym.spaces.Dict(new_ctx_spaces)})
        else:
            sub: gym.spaces.Box = ctx_space
            low  = np.array(sub.low,  dtype=np.float32).flatten()
            high = np.array(sub.high, dtype=np.float32).flatten()
            order_from_env = getattr(self.env, "ctx_to_observe", None)
            self._ctx_keys = [f"k{i}" for i in range(len(low))] if order_from_env is None else list(order_from_env)
            if len(self._ctx_keys) != len(low): raise ValueError("ctx_to_observe length mismatch")
            mids, halves = [], []
            for i, k in enumerate(self._ctx_keys):
                lo_i, hi_i = self.ranges.get(k, (float(low[i]), float(high[i])))
                mids.append(0.5*(lo_i+hi_i)); halves.append(max(1e-8, 0.5*(hi_i-lo_i)))
            self._mid_vec = np.array(mids, dtype=np.float32)
            self._half_vec = np.array(halves, dtype=np.float32)
            new_low  = np.full_like(low,  -1.0, dtype=np.float32)
            new_high = np.full_like(high, +1.0, dtype=np.float32)
            self.observation_space = gym.spaces.Dict({**{k: v for k, v in obs_space.spaces.items() if k != "context"},
                                                      "context": gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)})

    def observation(self, obs):
        obs = dict(obs); ctx = obs["context"]
        if self._is_dict_ctx:
            ctx_out = {}
            for k in self._ctx_keys:
                x = np.asarray(ctx[k], dtype=np.float32).reshape(-1)
                y = (x - self._mid_per_key[k]) / self._half_per_key[k]
                ctx_out[k] = np.clip(y, -5.0, 5.0).astype(np.float32)
            obs["context"] = ctx_out
        else:
            x = np.asarray(ctx, dtype=np.float32).reshape(-1)
            y = (x - self._mid_vec) / self._half_vec
            obs["context"] = np.clip(y, -5.0, 5.0).astype(np.float32)
        return obs

# =========================
# Utility / Seeding
# =========================
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)

# =============================================================================
# CHANGED: Dynamic Factory - Uses Actuators and Universal Schedulers
# =============================================================================
def dyn_factory_from_condition(ctx_key: str, cond_spec: Dict[str, Any], train_bounds: Tuple[float, float], payload: Dict[str, float]):
    kind = cond_spec["dyn_kind"]
    params = dict(cond_spec.get("dyn_params", {}))
    dseed  = int(cond_spec.get("dyn_seed", 0))
    lo, hi = map(float, train_bounds)

    if kind in ("identity", "none", "static"):
        return lambda worker_id: {}

    # --- Helper to create Actuator ---
    def make_act():
        return WalkerComXActuator(payload["mass"], payload["radius"], payload["y"])

    if kind in ("sin", "sine", "sinusoidal"):
        amp    = float(params.get("amplitude", 0.40))
        period = int(params.get("period", 200))
        def factory(worker_id: int):
            return {
                ctx_key: make_sinusoidal(
                    attribute=make_act(),  # Pass the Actuator object!
                    amplitude=amp, period=period,
                    min_val=lo, max_val=hi,
                    seed=dseed + worker_id,
                )
            }
        return factory

    # --- PIECEWISE CONSTANT ---
    if kind in ("piecewise_constant", "pwc"):
        values = params.get("values", [-0.5, -0.25, 0.0, 0.25, 0.5])
        int_range = params.get("interval_range", [200, 300])
        lo_i, hi_i = int(int_range[0]), int(int_range[1])
        if hi_i < lo_i: lo_i = hi_i

        def factory(worker_id: int):
            return {
                ctx_key: make_piecewise_constant(
                    attribute=make_act(),
                    values=[float(np.clip(v, lo, hi)) for v in values],
                    interval_range=(lo_i, hi_i),
                    min_val=lo, max_val=hi,
                    seed=dseed + worker_id,
                )
            }
        return factory

    # --- CONTINUOUS INCREMENTER ---
    if kind in ("continuous_incrementer", "ci"):
        span = (hi - lo)
        delta = float(params.get("delta", 0.001)) * span
        edge_mode  = str(params.get("edge_mode", "clip")).lower()
        ep_dir     = str(params.get("episode_direction", "random")).lower()
        base_dir   = str(params.get("direction", "both")).lower()
        keep_p     = float(params.get("follow_predefined_prob", 0.8))

        def factory(worker_id: int):
            return {
                ctx_key: make_continuous_incrementer(
                    attribute=make_act(),
                    delta=delta,
                    min_val=lo, max_val=hi,
                    edge_mode=edge_mode,
                    episode_direction=ep_dir,
                    direction=("both" if base_dir not in {"positive","negative"} else base_dir),
                    follow_predefined_prob=keep_p,
                    seed=dseed + worker_id,
                )
            }
        return factory

    # --- COSINE ANNEALING ---
    if kind in ("cosine_annealing", "ca"):
        ca_kwargs = {"mode": "cycle"}
        for k in ("T_0", "T_mult", "retarget", "neighborhood_radius"):
            if k in params:
                ca_kwargs[k] = params[k]

        def factory(worker_id: int):
            return {
                ctx_key: make_cosine_annealing(
                    attribute=make_act(),
                    min_val=lo, max_val=hi,
                    seed=dseed + worker_id,
                    **ca_kwargs
                )
            }
        return factory


    if kind in ("random_walk", "rw"):
        std = float(params.get("std", 0.05))
        def factory(worker_id: int):
            return {
                ctx_key: make_random_walk(
                    attribute=make_act(),
                    std=std,
                    min_val=lo, max_val=hi,
                    seed=dseed + worker_id,
                )
            }
        return factory
    
    # --- SUDDEN JUMP ---
    if kind in ("sudden_jump", "jump"):
         step_range = params.get("step_size_range", (0.2, 0.4))
         int_range = params.get("interval_range", (10, 30))
         
         def factory(worker_id: int):
            return {
                ctx_key: make_sudden_jump(
                    attribute=make_act(),
                    step_size_range=step_range,
                    interval_range=int_range,
                    min_val=lo, max_val=hi,
                    seed=dseed + worker_id,
                    direction="both",
                    edge_mode="clip"
                )
            }
         return factory

    raise ValueError(f"Unknown dyn_kind='{kind}'")

# =========================
# Env builders (Walker)
# =========================
def make_walker_training_env(
    *,
    train_ctxs: Dict[int, Dict[str,float]],
    dyn_factory: Callable[[int], Dict[str, Any]],
    payload: Dict[str, float],
    observe_ctx: bool,
    keep_center_when_static: bool,
    context_norm_ranges: Dict[str, tuple[float,float]] | None,
    seed: int,
    worker_id: int = 0,
    monitor_path: Path | None = None,
    feature_bounds: Tuple[float, float] | None = None,   # NEW: declared range clamp
) -> gym.Env:

    # attach normalizer only if we're actually observing context and it exists
    def _has_nonempty_context(space):
        if not (isinstance(space, gym.spaces.Dict) and "context" in space.spaces):
            return False
        ctx_space = space.spaces["context"]
        if isinstance(ctx_space, gym.spaces.Dict):
            return len(ctx_space.spaces) > 0
        if isinstance(ctx_space, gym.spaces.Box):
            return ctx_space.shape is not None and np.prod(ctx_space.shape) > 0
        return False

    use_dyn = (dyn_factory is not None and dyn_factory(0) != {})

    base = walker_env_factory(
        render_mode=None,
        contexts=train_ctxs,
        ctx_to_observe=(["COM_X"] if observe_ctx else []),
        payloaded=True,
        keep_center=(keep_center_when_static and not use_dyn),
        payload_kwargs=payload,
    )

    lo, hi = feature_bounds if feature_bounds is not None else (-0.6, +0.6)

    env = GymDynamicContextCarlWrapper(
        base,
        feature_update_fns=(dyn_factory(worker_id) if dyn_factory else {}),
        ctx_getters={"COM_X": _get_hull_com_x},
        ctx_setters={"COM_X": (lambda e, v: _set_hull_com_x(
            e, x=float(v), mass=payload["mass"], radius=payload["radius"], y_fixed=payload["y"]
        ))},
        ctx_to_observe=(["COM_X"] if observe_ctx else []),
        observe_context_mode=("live" if observe_ctx else "none"),
        mutate_obs_space=False,
        worker_id=worker_id,
        verbose=False,
        dctx_features_definitions={
            "COM_X": UniformFloatContextFeature("COM_X", lower=float(lo), upper=float(hi), default_value=0.0)
        },
    )

    # Normalize CONTEXT ONLY (still fine when context dict is empty)
    if context_norm_ranges:
        env = ContextOnlyNormalize(env, context_norm_ranges)

    # If 'context' space exists but is EMPTY (obs_mode='none'), wrap so flatten won't crash
    if isinstance(env.observation_space, gym.spaces.Dict) and "context" in env.observation_space.spaces:
        ctx_space = env.observation_space.spaces["context"]
        is_empty_ctx = isinstance(ctx_space, gym.spaces.Dict) and len(ctx_space.spaces) == 0
        if is_empty_ctx:
            env = PartialContextWrapper(env, selected_context_keys=[])

    # Safe to flatten now
    env = FlattenObservation(env)

    if monitor_path:
        env = Monitor(env, filename=str(monitor_path), allow_early_resets=True)

    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    return env

# =========================
# PPO factory (simple)
# =========================
def make_ppo(env, *, total_steps: int, rollout_steps: int, tensorboard_log: Path | None, device: str = "cpu"):
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_envs = getattr(env, "num_envs", 1)
    per_env_steps = max(1, int(rollout_steps) // int(n_envs))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=per_env_steps,
        device=device,
        verbose=1,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )
    return model

# =========================
# Per-Context eval callback (unchanged)
# =========================
class PerContextEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        tag: str,
        eval_ctxs: Dict[int, Dict[str, float]],
        make_env_for_ctx: Callable[[Dict[str,float]], SubprocVecEnv],
        eval_freq: int = 20_000,
        n_episodes_per_ctx: int = 24,
        deterministic: bool = True,
        seed_summary_dir: Path,
        eval_n_envs: int = 4,
    ):
        super().__init__(verbose=0)
        self.tag = tag
        self.eval_ctxs = eval_ctxs
        self.make_env_for_ctx = make_env_for_ctx
        self.eval_freq = int(eval_freq)
        self.n_episodes_per_ctx = int(n_episodes_per_ctx)
        self.deterministic = bool(deterministic)
        self.eval_n_envs = int(eval_n_envs)
        self.seed_summary_dir = Path(seed_summary_dir)
        self._history: List[Tuple[int, Dict[str, float]]] = []
        self.best_avg = -np.inf
        self.best_min = -np.inf
        self.best_step = -1

        self._vecenv_per_ctxname: Dict[str, Any] = {}
        for k, v in self.eval_ctxs.items():
            name = f"COMX_{float(v['COM_X']):+.2f}"
            self._vecenv_per_ctxname[name] = self.make_env_for_ctx(v)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True
        means = {}
        for name, venv in self._vecenv_per_ctxname.items():
            n_eps = int(np.ceil(self.n_episodes_per_ctx / self.eval_n_envs) * self.eval_n_envs)
            t0 = time.time()
            ep_rewards, _ = evaluate_policy(
                self.model, venv,
                n_eval_episodes=n_eps,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
            )
            dt = time.time() - t0
            mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            std_r  = float(np.std(ep_rewards))  if ep_rewards else 0.0
            self.logger.record(f"eval_ctx_reward_{self.tag}/{name}/mean_reward", mean_r)
            self.logger.record(f"eval_ctx_std_reward_{self.tag}/{name}/std_reward", std_r)
            self.logger.record(f"eval_timing_{self.tag}/{name}/seconds", float(dt))
            means[name] = mean_r

        if means:
            vals = list(means.values())
            agg_avg = float(np.mean(vals))
            agg_min = float(np.min(vals))
            self.logger.record(f"eval_reward_aggregate/{self.tag}/avg_reward_ood", agg_avg)
            self.logger.record(f"eval_reward_aggregate/{self.tag}/min_reward_ood", agg_min)
            if agg_avg > self.best_avg:
                self.best_avg = agg_avg
                self.best_min = agg_min
                self.best_step = int(self.num_timesteps)
            self._history.append((int(self.num_timesteps), means))
        return True

    def _on_training_end(self) -> None:
        self.seed_summary_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.seed_summary_dir / f"seed_summary_{self.tag}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "metric", "value"])
            w.writerow([self.best_step, "best_robust_avg", self.best_avg])
            w.writerow([self.best_step, "best_robust_min", self.best_min])
            if self._history:
                step, last_means = self._history[-1]
                for ctx_name, val in last_means.items():
                    w.writerow([step, f"last_{ctx_name}_mean_reward", val])
        for v in self._vecenv_per_ctxname.values():
            try: v.close()
            except Exception: pass

# =========================
# Main (macro-driven; no argparse)
# =========================
def main():
    bits = load_from_macro_env_and_yaml()
    cfg = bits["cfg"]
    ctx_key = bits["ctx_key"]
    seed = bits["seed"]
    eval_n_envs = bits["eval_n_envs"]
    eval_base_seed = bits["eval_base_seed"]
    train_lo, train_hi = bits["train_bounds"]
    eval_feature_bounds = bits["eval_feature_bounds"]
    train_vals = bits["train_vals"]
    eval_vals = bits["eval_vals"]
    context_norm_ranges = bits["context_norm_ranges"]
    pool_name = bits["pool_name"]
    cond_name = bits["cond_name"]
    cond_spec = bits["cond_spec"]

    seed_everything(seed)

    # --- dirs (compact & unambiguous) ---
    results_root = Path(
        os.environ.get("RESULTS_ROOT")                                # <- prefer cluster-provided root
        or cfg.get("results_root")
        or (project_root / "dynamic_crl" / "final_thesis_experiments" / "walker_quick_try" / "runs")
    )
    print(f"[paths] results_root = {results_root}")

    tkind = cond_spec["train_ctxs_kind"]  # "single" | "pool"
    run_dir, tb_dir, ckpt_dir, models_dir = _exp_path(
        results_root, ctx_key, cond_name, tkind, (pool_name if tkind != "single" else None), seed
    )
    for d in (tb_dir, ckpt_dir, models_dir, run_dir / "summaries"):
        d.mkdir(parents=True, exist_ok=True)

    # --- payload ---
    payload: Dict[str, float] = dict(cfg.get("PAYLOAD") or {"mass": 1.5, "radius": 0.4, "y": 0.0})

    # --- dynamic factory strictly within train_ranges ---
    dyn_factory = dyn_factory_from_condition(ctx_key, cond_spec, (train_lo, train_hi), payload)

    # --- TRAIN env ---
    observe_ctx_train = (bits["obs_mode"] == "live")
    observe_ctx_eval  = observe_ctx_train

    train_ctxs = {i: {ctx_key: float(v)} for i, v in enumerate(train_vals)}
    def _make_train():
        return make_walker_training_env(
            train_ctxs=train_ctxs,
            dyn_factory=dyn_factory,
            payload=payload,
            observe_ctx=observe_ctx_train,         # honor obs_mode
            keep_center_when_static=True,
            context_norm_ranges=context_norm_ranges,
            seed=seed,
            worker_id=0,
            monitor_path=run_dir / f"monitor_seed_{seed}",
            feature_bounds=(train_lo, train_hi),
        )
    env = DummyVecEnv([_make_train])

    # --- PPO ---
    ppo_cfg = (cfg.get("ppo") or {})
    total_steps   = int(ppo_cfg.get("total_steps", 150_000))
    rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
    model = make_ppo(
        env, total_steps=total_steps, rollout_steps=rollout_steps,
        tensorboard_log=tb_dir, device="auto"
    )

    eval_freq = int(ppo_cfg.get("eval_freq", 100_000))

    short_cond = _shorten_cond_name(cond_name)
    ckpt_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(ckpt_dir),
        name_prefix=f"ckpt_{ctx_key.lower()}__{short_cond}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # --- Static-eval builders (obs matches training) ---
    def make_static_eval_env_for_ctx(ctx_dict: Dict[str, float]) -> SubprocVecEnv:
        val = float(ctx_dict[ctx_key])
        def _one(rank: int):
            def _f():
                one_ctx = {0: {ctx_key: val}}
                return make_walker_training_env(
                    train_ctxs=one_ctx,
                    dyn_factory=(lambda _: {}),  # static eval
                    payload=payload,
                    observe_ctx=observe_ctx_eval,
                    keep_center_when_static=True,
                    context_norm_ranges=context_norm_ranges,
                    seed=eval_base_seed + rank,
                    worker_id=rank,
                    monitor_path=None,
                    feature_bounds=eval_feature_bounds,  # allow true OOD during eval
                )
            return _f
        return SubprocVecEnv([_one(r) for r in range(eval_n_envs)], start_method="spawn")

    per_seed_summary_dir = run_dir / "summaries"
    cb_low = PerContextEvalCallback(
        tag="ood_low",
        eval_ctxs={i: {ctx_key: float(v)} for i, v in enumerate(eval_vals["ood_low"])},
        make_env_for_ctx=make_static_eval_env_for_ctx,
        eval_freq=eval_freq, n_episodes_per_ctx=24, deterministic=True,
        seed_summary_dir=per_seed_summary_dir, eval_n_envs=eval_n_envs,
    )
    cb_high = PerContextEvalCallback(
        tag="ood_high",
        eval_ctxs={i: {ctx_key: float(v)} for i, v in enumerate(eval_vals["ood_high"])},
        make_env_for_ctx=make_static_eval_env_for_ctx,
        eval_freq=eval_freq, n_episodes_per_ctx=24, deterministic=True,
        seed_summary_dir=per_seed_summary_dir, eval_n_envs=eval_n_envs,
    )
    cb_id = PerContextEvalCallback(
        tag="id",
        eval_ctxs={i: {ctx_key: float(v)} for i, v in enumerate(eval_vals["id"])},
        make_env_for_ctx=make_static_eval_env_for_ctx,
        eval_freq=eval_freq, n_episodes_per_ctx=24, deterministic=True,
        seed_summary_dir=per_seed_summary_dir, eval_n_envs=eval_n_envs,
    )
    callbacks = CallbackList([ckpt_cb, cb_low, cb_high, cb_id])

    # --- learn (short TB run name; folder encodes condition) ---
    tb_name = f"seed_{seed}"
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        tb_log_name=tb_name,
        progress_bar=False,
    )

    # --- save + meta ---
    final_path = models_dir / f"{ctx_key.lower()}_ppo_final_{total_steps//1000}k_seed{seed}.zip"
    model.save(str(final_path))

    with open(run_dir / "run_meta.json", "w") as f:
        json.dump({
            "cond_name": cond_name,
            "cond_spec": cond_spec,
            "ctx_key": ctx_key,
            "pool_name": (pool_name if tkind != "single" else None),
            "train_kind": tkind,
            "obs_mode_train": observe_ctx_train,
            "obs_mode_eval": observe_ctx_eval,
            "train_bounds": [train_lo, train_hi],
            "eval_feature_bounds": list(eval_feature_bounds),
            "seed": seed,
            "eval_n_envs": eval_n_envs,
            "rollout_steps": rollout_steps,
            "total_steps": total_steps,
        }, f, indent=2)

    print(f"[seed {seed}] Saved: {final_path}")
    try:
        env.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()