#!/usr/bin/env python3
# cp_exploration_speed_runner.py
# -------------------------------------------------------
# Compare exploration speed for three setups over a 7-context pool:
#   1) dynamic_single:     single env with in-episode dynamics
#   2) static_parallel:    parallel envs with static per-episode context resampling
#   3) static_sequential:  single env with static per-episode resampling
#
# Each run logs:
#   - TB, checkpoints
#   - state/context series via GymStateContextLogger
#   - a per-step global timeline via GlobalTickTracker (global_tick.csv)
#
# Fair-time comparison rule:
#   - One "global tick" corresponds to one synchronous step across all parallel envs.
#   - For single-env runs, the same CSV still logs global_tick = raw env step.

from __future__ import annotations
import os, sys, json, csv
from pathlib import Path
from typing import Any, Dict, Sequence, Callable, List, Optional, Tuple

import numpy as np
import yaml
import time
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch.utils.tensorboard import SummaryWriter

# --- project path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(os.path.abspath(os.path.join(current_dir, '../../../')))
sys.path.append(str(project_root))

# --- your modules ---
from dynamic_crl.src.utils.gym_utils import cartpole_env_factory
from dynamic_crl.src.dynamic_carl.env_wrappers import GymDynamicContextCarlWrapper
from dynamic_crl.src.dynamic_carl.state_context_tracker import GymStateContextLogger

from dynamic_crl.src.utils.gym_utils import get_gym_base_env, _make_cp_ctx_accessor
from dynamic_crl.src.utils.log_msgs import info_msg, warn_msg, error_msg, success_msg
from dynamic_crl.src.utils.utils import load_cfg, seed_everything
from dynamic_crl.helix_utils.helix_submitit import resolve_run_root

from dynamic_crl.src.dynamic_carl.scheduler_utils import build_update_fn_from_spec
from dynamic_crl.src.dynamic_carl.gym_context_updates import with_masspole_from_length

os.environ["SDL_VIDEODRIVER"] = "dummy"

CTX_KEY = "length"

_len_get, _len_set = _make_cp_ctx_accessor("length")
_for_get, _for_set = _make_cp_ctx_accessor("force_mag")
_gra_get, _gra_set = _make_cp_ctx_accessor("gravity")

CTX_REGISTRY = {
    #"length":   {"getter": _len_get, "setter": _len_set, "postprocess_updater": with_masspole_from_length, "nice_name": "L"},
    # we are NOT using masspole coupling here to match CARL's ctx sampling logic (masspole is contstant w.r.t length changes)
    "length":   {"getter": _len_get, "setter": _len_set, "postprocess_updater": (lambda u: u), "nice_name": "L"},
    "force_mag":{"getter": _for_get, "setter": _for_set, "postprocess_updater": (lambda u: u), "nice_name": "F"},
    "gravity":  {"getter": _gra_get, "setter": _gra_set, "postprocess_updater": (lambda u: u), "nice_name": "G"},
}

# ----------------------
# Dynamic context update functions (defaults)
# ----------------------
SIN_AMPLITUDE = 0.2
SIN_PERIOD    = 500
LEN_MIN       = 0.35
LEN_MAX       = 0.75

# Whether to use the "best schedulers" section (dynamic_best) or the default (dynamic)
USE_BEST = os.environ.get("EXPL_USE_BEST", "0").strip().lower() in ("1", "true", "yes")

# When USE_BEST is enabled, selects which entry inside dynamic_best to use.
# Example: "best_continuous_incrementer_pool7" or "best_cosine_annealing_pool7"
DYN_NAME = os.environ.get("DYN_NAME", None)

# --- loading the dynamics from the config ---
def _find_dynamic_by_name(cfg: Dict[str, Any], section: str, target_name: str) -> Dict[str, Any]:
    """
    Look up a dynamic spec by name in cfg[section].
    section is either "dynamic" (default sinusoidal presets)
    or "dynamic_best" (hand-picked best schedulers).
    """
    dyns = cfg.get(section, []) or []
    for d in dyns:
        if d.get("name") == target_name:
            return d
    raise ValueError(
        f"Dynamic spec named '{target_name}' not found in YAML.{section}"
    )

def _dynamic_spec_for_pool(
    cfg: Dict[str, Any],
    pool_name: str,
    pool_vals: Sequence[float],
) -> Dict[str, Any]:
    """
    Decide which dynamic spec to use and return the *full* spec dict.

    Two modes, driven only by EXPL_USE_BEST:

    1) USE_BEST == False (normal runs):
         - use cfg["dynamic"] and the original pool-size → sinusoidal mapping:
             * len(pool_vals) == 1  -> sinusoidal_large
             * len(pool_vals) == 3  -> sinusoidal_medium
             * len(pool_vals) == 7  -> sinusoidal_small
           with a name-based fallback.

    2) USE_BEST == True (best-scheduler runs):
         - ignore pool_name & pool_vals (your "best" config YAML already only
           contains pool7 etc.).
         - require DYN_NAME to be set and look up that spec in cfg["dynamic_best"].
    """
    # --- Best-mode: explicit selection from dynamic_best ---
    if USE_BEST:
        section = "dynamic_best"
        if not DYN_NAME:
            raise ValueError(
                "EXPL_USE_BEST is set, but DYN_NAME is not provided. "
                "Please set DYN_NAME to one of the entries in 'dynamic_best', e.g. "
                "DYN_NAME=best_continuous_incrementer_pool7 or "
                "DYN_NAME=best_cosine_annealing_pool7."
            )
        return _find_dynamic_by_name(cfg, section, DYN_NAME)

    # --- Default mode: original sinusoidal mapping by pool size (cfg['dynamic']) ---
    section = "dynamic"
    n = len(pool_vals)
    pn = str(pool_name).lower()

    if   n == 1:
        target = "sinusoidal_large"
    elif n == 3:
        target = "sinusoidal_medium"
    elif n == 7:
        target = "sinusoidal_small"
    else:
        # name-based fallback (keeps working if you add extra pools later)
        if   pn.startswith("single_"):
            target = "sinusoidal_large"
        elif "pool3" in pn:
            target = "sinusoidal_medium"
        elif "pool7" in pn:
            target = "sinusoidal_small"
        else:
            raise ValueError(
                f"Don't know which dynamic preset to use for pool '{pool_name}' "
                f"with {n} values (USE_BEST={USE_BEST})."
            )

    return _find_dynamic_by_name(cfg, section, target)


# ----------------------
# State/Context logger attach
# ----------------------
def attach_state_context_logger(env: gym.Env, *, out_root: Path, worker_id: int,
                                ctx_keys: Sequence[str], obs_mode: Optional[str]) -> gym.Env:
    logs_dir = out_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def true_ctx_getter(raw_env, obs, info):
        base = get_gym_base_env(raw_env)
        if hasattr(base, "length"):
            try:
                return {"length": float(getattr(base, "length"))}
            except Exception:
                return None
        try:
            L = env.get_wrapper_attr("length")
            if L is not None:
                return {"length": float(L)}
        except Exception:
            pass
        return None

    return GymStateContextLogger(
        env,
        true_context_keys=list(ctx_keys),
        observation_mode=obs_mode,      # "initial" | "live" | None
        context_getter=true_ctx_getter,
        state_mode="exact",             # keep exact; bin later in analysis
        context_mode="exact",
        round_ndigits=8,
        out_state_csv=str(logs_dir / "state_series.csv"),
        out_true_context_csv=str(logs_dir / "true_context_series.csv"),
        out_observed_context_csv=str(logs_dir / "observed_context_series.csv"),
        env_id=worker_id,
        split_csv_per_env=True,
        overwrite_logs=True,
        log_observed_contexts=list(ctx_keys),
        verbose=True,
    )

# ----------------------
# Global tick tracker (VecEnv wrapper)
# ----------------------
class GlobalTickTracker(VecEnvWrapper):
    """
    Wraps any VecEnv and appends one CSV row per synchronous collector step:
      global_tick, env_id, episode, done
    (Episodes are env-local counters inferred from dones.)
    """
    def __init__(self, venv, log_csv_path: Path):
        super().__init__(venv, venv.observation_space, venv.action_space)
        self.global_tick = 0
        self.ep_ids = [0 for _ in range(self.num_envs)]
        self.csv_path = Path(log_csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._w = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._w)
        self._writer.writerow(["global_tick", "env_id", "episode", "done"])
        self._w.flush()

    def reset(self):
        obs = self.venv.reset()
        self.ep_ids = [0 for _ in range(self.num_envs)]
        return obs

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.global_tick += 1
        for i, d in enumerate(dones):
            self._writer.writerow([self.global_tick, i, self.ep_ids[i], bool(d)])
            if d:
                self.ep_ids[i] += 1
        self._w.flush()
        return obs, rewards, dones, infos

    def close(self):
        try:
            self.venv.close()
        finally:
            try:
                self._w.close()
            except Exception:
                pass

# ----------------------
# Eval helper: fixed per-episode seed schedule
# ----------------------
class EvalResetSeedWrapper(gym.Wrapper):
    def __init__(self, env, seed_schedule: List[int]):
        super().__init__(env)
        self._seeds = [int(s) for s in seed_schedule]
        self._i = 0

    def reset(self, **kwargs):
        if self._i < len(self._seeds):
            kwargs = dict(kwargs or {})
            kwargs["seed"] = self._seeds[self._i]
            self._i += 1
        return self.env.reset(**kwargs)

    # NEW: allow resetting the per-episode seed pointer
    def rewind_eval_seeds(self):
        self._i = 0

# ----------------------
# Env builders with state observations
# ----------------------
def make_static_env(
    *, seed: int, out_root: Path, worker_id: int, contexts: Dict[int, Dict[str, float]],
    observe_mode: str = "live", monitor: bool = True
) -> gym.Env:
    env = cartpole_env_factory(contexts=contexts, render_mode=None, ctx_to_observe=[CTX_KEY])

    env = GymDynamicContextCarlWrapper(
        env=env,
        feature_update_fns={},               # STATIC
        ctx_getters={"length": _len_get},
        ctx_setters={"length": _len_set},
        ctx_to_observe=[CTX_KEY],
        worker_id=worker_id,
        observe_context_mode=observe_mode,   # "live" usually for exposure during training
        mutate_obs_space=False,
        verbose=False,
    )

    env = attach_state_context_logger(env, out_root=out_root, worker_id=worker_id, ctx_keys=[CTX_KEY], obs_mode=observe_mode)
    env = FlattenObservation(env)

    if monitor:
        env = Monitor(env, filename=str(out_root / f"monitor_seed_{seed}_wid{worker_id}"), allow_early_resets=True)

    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    return env

def make_dynamic_env(
    *, seed: int, out_root: Path, worker_id: int,
    contexts: Dict[int, Dict[str, float]],
    observe_mode: str = "live",
    feature_update_fns: Optional[Dict[str, Callable[..., Any]]] = None,
    monitor: bool = True
) -> gym.Env:
    """
    Dynamic single-env: pass your dynamic context update fns via `feature_update_fns`.
    If None, we default to STATIC (no updates), which you should replace with your own.
    """
    env = cartpole_env_factory(contexts=contexts, render_mode=None, ctx_to_observe=[CTX_KEY])

    env = GymDynamicContextCarlWrapper(
        env=env,
        feature_update_fns=feature_update_fns or {},
        ctx_getters={"length": _len_get},
        ctx_setters={"length": _len_set},
        ctx_to_observe=[CTX_KEY],
        worker_id=worker_id,
        observe_context_mode=observe_mode,
        mutate_obs_space=False,
        verbose=False,
    )

    env = attach_state_context_logger(env, out_root=out_root, worker_id=worker_id, ctx_keys=[CTX_KEY], obs_mode=observe_mode)
    env = FlattenObservation(env)

    if monitor:
        env = Monitor(env, filename=str(out_root / f"monitor_seed_{seed}_wid{worker_id}"), allow_early_resets=True)

    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    return env

# ----------------------
# Eval vec builder (STATIC eval + deterministic resets)
# ----------------------
def make_eval_vec_for_ctx(
    *, ctx_dict: Dict[str, Any],  # e.g. {"length": 0.55}
    eval_n_envs: int,
    out_root: Path,
    base_seed: int,
    observe_mode: str = "live",
) -> SubprocVecEnv:
    """
    Builds a vectorized STATIC eval env with a fixed per-episode seed schedule.
    Deterministic across reruns for the same base_seed.
    """
    def _make_one(rank: int):
        def _f():
            worker_seed = int(base_seed) + int(rank)

            env = cartpole_env_factory(contexts={0: ctx_dict}, render_mode=None, ctx_to_observe=[CTX_KEY])
            env = GymDynamicContextCarlWrapper(
                env=env,
                feature_update_fns={},               # STATIC eval
                ctx_getters={"length": _len_get},
                ctx_setters={"length": _len_set},
                ctx_to_observe=[CTX_KEY],
                worker_id=rank,
                observe_context_mode=observe_mode,   # keep LIVE to match training obs
                mutate_obs_space=False,
                verbose=False,
            )
            env = FlattenObservation(env)

            # Deterministic per-episode resets: unique schedule per rank
            # Large schedule to cover any n_eval_episodes requested
            n_needed = 10000
            seed_schedule = [base_seed + 10_000 * rank + i for i in range(n_needed)]
            env = EvalResetSeedWrapper(env, seed_schedule)

            try:
                env.reset(seed=worker_seed)  # initial reset
            except TypeError:
                pass
            return env
        return _f

    return SubprocVecEnv([_make_one(r) for r in range(eval_n_envs)], start_method="spawn")

def make_env_for_ctx_factory(*, eval_n_envs: int, out_root: Path, base_seed: int):
    def _make_env_for_ctx(ctx_dict: Dict[str, Any], context_mode: str = "concat_ctx"):
        # context_mode is accepted to satisfy the callback signature; not used here.
        return make_eval_vec_for_ctx(
            ctx_dict=ctx_dict,
            eval_n_envs=eval_n_envs,
            out_root=out_root,
            base_seed=base_seed,
            observe_mode="live",
        )
    return _make_env_for_ctx

# ----------------------
# Per-context eval callback (unchanged logic, but envs are now deterministic)
# ----------------------
class PerContextEvalCallback(BaseCallback):
    """
    Static eval per context split ('lo', 'hi', 'id'). Writes seed_summary_<tag>.csv and logs TB scalars.
    """
    def __init__(
        self, ctx_key: str, make_env_for_ctx: Callable[[Dict[str, Any], str], Any],
        eval_ctxs: Dict[Any, Dict[str, Any]], eval_freq: int = 10_000, n_episodes_per_ctx: int = 30,
        deterministic: bool = True, seed_summary_dir: Path | None = None, verbose: int = 0,
        context_mode: str = "concat_ctx", tag: str = "static", eval_n_envs: int = 1,
    ):
        super().__init__(verbose)
        self.ctx_key = ctx_key
        self.make_env_for_ctx = make_env_for_ctx
        self.eval_ctxs = eval_ctxs
        self.eval_freq = eval_freq
        self.n_episodes_per_ctx = n_episodes_per_ctx
        self.deterministic = deterministic
        self.seed_summary_dir = Path(seed_summary_dir) if seed_summary_dir else None
        self.tag = tag
        self.eval_n_envs = eval_n_envs
        self.ctx_envs = {
            self._ctx_name(k, v): self.make_env_for_ctx(v, context_mode=context_mode)
            for k, v in self.eval_ctxs.items()
        }
        self._history: List[Tuple[int, Dict[str, float]]] = []
        self.best_robust_avg = -np.inf
        self.best_robust_min = -np.inf
        self.best_step = -1

    def _ctx_name(self, k, v):
        if self.ctx_key in v:
            nn = CTX_REGISTRY[self.ctx_key]["nice_name"]
            return f"{nn}_{float(v[self.ctx_key]):.3f}"
        return f"ctx_{k}"

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        # Rewind all eval envs so each checkpoint evaluates on the SAME episodes
        for env in self.ctx_envs.values():
            try:
                env.env_method("rewind_eval_seeds")  # call on all ranks in SubprocVecEnv
                env.reset()                          # optional: align initial reset
            except Exception:
                pass

        means = {}
        for name, env in self.ctx_envs.items():
            n_eps = int(np.ceil(self.n_episodes_per_ctx / self.eval_n_envs) * self.eval_n_envs)
            t0 = time.time()
            ep_rewards, _ = evaluate_policy(
                self.model, env,
                n_eval_episodes=n_eps,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
            )
            dt = time.time() - t0
            mean_r = float(np.mean(ep_rewards)) if len(ep_rewards) else 0.0
            std_r  = float(np.std(ep_rewards))  if len(ep_rewards) else 0.0
            self.logger.record(f"eval_ctx_reward_{self.tag}/{name}/mean_reward", mean_r)
            self.logger.record(f"eval_ctx_std_reward_{self.tag}/{name}/std_reward",  std_r)
            self.logger.record(f"eval_timing_{self.tag}/{name}/seconds", float(dt))
            means[name] = mean_r

        if means:
            robust_avg = float(np.mean(list(means.values())))
            robust_min = float(np.min(list(means.values())))
            self.logger.record(f"eval_reward_aggregate/{self.tag}/avg_reward_ood", robust_avg)
            self.logger.record(f"eval_reward_aggregate/{self.tag}/min_reward_ood", robust_min)
            if robust_avg > self.best_robust_avg:
                self.best_robust_avg = robust_avg
                self.best_robust_min = robust_min
                self.best_step = int(self.num_timesteps)
            self._history.append((int(self.num_timesteps), means))
        return True

    def _on_training_end(self) -> None:
        if self.seed_summary_dir is None:
            return
        self.seed_summary_dir.mkdir(parents=True, exist_ok=True)
        last_means = self._history[-1][1] if len(self._history) else {}
        csv_path = self.seed_summary_dir / f"seed_summary_{self.tag}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "metric", "value"])
            w.writerow([self.best_step, "best_robust_avg", self.best_robust_avg])
            w.writerow([self.best_step, "best_robust_min", self.best_robust_min])
            for ctx, val in last_means.items():
                w.writerow([self._history[-1][0] if self._history else -1, f"last_{ctx}_mean_reward", val])
        for _env in self.ctx_envs.values():
            try: _env.close()
            except Exception: pass

# ----------------------
# Parallelism utils
# ----------------------
def _contexts_for_rank(pool_vals: Sequence[float], rank: int, *, mode: str = "rotate") -> Dict[int, Dict[str, float]]:
    """
    Deterministically permute the pool per worker.
    mode="rotate": rotate left by 'rank' positions.
    """
    vals = list(map(float, pool_vals))
    if not vals:
        raise ValueError("Empty pool_vals")

    if mode == "rotate":
        k = rank % len(vals)
        perm = vals[k:] + vals[:k]
    else:
        perm = vals

    ctx = {i: {CTX_KEY: v} for i, v in enumerate(perm)}
    if len(perm) < 2:
        ctx[1] = {CTX_KEY: float(perm[0])}
    return ctx

# ----------------------
# Build vec envs for each condition
# ----------------------
def build_vec_for_condition(
    condition: str,
    *,
    pool_vals: Sequence[float],
    seed: int,
    out_root: Path,
    n_envs_parallel: int,
    feature_update_fns: Optional[Dict[str, Callable[..., Any]]] = None,
    parallel_ctx_perm: str = "rotate",
):
    """
    Returns (vec_env, manifest_dict).
    Supported conditions:
      - "static_sequential"  : 1 env, no dynamics
      - "dynamic_sequential" : 1 env, with dynamics (feature_update_fns)
      - "static_parallel"    : n_envs_parallel >= 2, no dynamics
      - "dynamic_parallel"   : n_envs_parallel >= 2, with dynamics (feature_update_fns)
    """
    context_dict = {i: {CTX_KEY: float(v)} for i, v in enumerate(pool_vals)}

    # Handle single-value pool by duplicating context ID 1
    if len(pool_vals) < 2 and pool_vals[0] is not None:
        context_dict[1] = {CTX_KEY: float(pool_vals[0])}

    if condition == "static_sequential":
        def _thunk():
            return make_static_env(
                contexts=context_dict, seed=seed, out_root=out_root, worker_id=0, observe_mode="live"
            )
        vec = DummyVecEnv([_thunk])
        manifest = dict(mode="static_sequential", n_envs=1, ctx_key=CTX_KEY, obs_mode="live",
                        dynamics=False, pool_vals=list(map(float, pool_vals)))

    elif condition == "extra_static_sequential":
        def _thunk():
            return make_static_env(
                contexts=context_dict, seed=seed, out_root=out_root, worker_id=0, observe_mode="live"
            )
        vec = DummyVecEnv([_thunk])
        manifest = dict(mode="extra_static_sequential", n_envs=1, ctx_key=CTX_KEY, obs_mode="live",
                        dynamics=False, pool_vals=list(map(float, pool_vals)))

    elif condition == "dynamic_sequential":
        def _thunk():
            return make_dynamic_env(
                seed=seed, out_root=out_root, worker_id=0,
                feature_update_fns=feature_update_fns or {},
                observe_mode="live", contexts=context_dict
            )
        vec = DummyVecEnv([_thunk])
        manifest = dict(mode="dynamic_sequential", n_envs=1, ctx_key=CTX_KEY, obs_mode="live",
                        dynamics=True, pool_vals=list(map(float, pool_vals)))

    elif condition == "static_parallel":
        assert n_envs_parallel >= 2, "static_parallel requires n_envs_parallel >= 2"
        def _mk(rank: int):
            def _f():
                context_dict_rank = _contexts_for_rank(pool_vals, rank, mode=parallel_ctx_perm)
                return make_static_env(
                    contexts=context_dict_rank, seed=seed + rank, out_root=out_root,
                    worker_id=rank, observe_mode="live"
                )
            return _f
        vec = SubprocVecEnv([_mk(i) for i in range(n_envs_parallel)], start_method="spawn")
        manifest = dict(mode="static_parallel", n_envs=n_envs_parallel, ctx_key=CTX_KEY, obs_mode="live",
                        dynamics=False, pool_vals=list(map(float, pool_vals)),
                        parallel_ctx_perm=parallel_ctx_perm)

    elif condition == "dynamic_parallel":
        assert n_envs_parallel >= 2, "dynamic_parallel requires n_envs_parallel >= 2"
        def _mk(rank: int):
            def _f():
                context_dict_rank = _contexts_for_rank(pool_vals, rank, mode=parallel_ctx_perm)
                return make_dynamic_env(
                    seed=seed + rank, out_root=out_root, worker_id=rank,
                    feature_update_fns=feature_update_fns or {},
                    observe_mode="live", contexts=context_dict_rank
                )
            return _f
        vec = SubprocVecEnv([_mk(i) for i in range(n_envs_parallel)], start_method="spawn")
        manifest = dict(mode="dynamic_parallel", n_envs=n_envs_parallel, ctx_key=CTX_KEY, obs_mode="live",
                        dynamics=True, pool_vals=list(map(float, pool_vals)),
                        parallel_ctx_perm=parallel_ctx_perm)

    elif condition == "extra_static_parallel":
        assert n_envs_parallel >= 2, "extra_static_parallel requires n_envs_parallel >= 2"
        def _mk(rank: int):
            def _f():
                context_dict_rank = _contexts_for_rank(pool_vals, rank, mode=parallel_ctx_perm)
                return make_static_env(
                    contexts=context_dict_rank, seed=seed + rank, out_root=out_root,
                    worker_id=rank, observe_mode="live"
                )
            return _f
        vec = SubprocVecEnv([_mk(i) for i in range(n_envs_parallel)], start_method="spawn")
        manifest = dict(mode="extra_static_parallel", n_envs=n_envs_parallel, ctx_key=CTX_KEY, obs_mode="live",
                        dynamics=False, pool_vals=list(map(float, pool_vals)),
                        parallel_ctx_perm=parallel_ctx_perm)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return vec, manifest

# ----------------------
# Main launcher
# ----------------------
if __name__ == "__main__":
    # Decide which base YAML to use.
    # Priority:
    #   1) EXPL_USE_BEST=1  → force best-sched config
    #   2) EXPL_CFG         → explicit path (only when not using best)
    #   3) default config

    # e.g usage to run for best schedulers:
    # EXPL_USE_BEST=1 \
    # DYN_NAME=best_continuous_incrementer_pool7 \
    # python cp_state_exploration_runner.py

    default_cfg = Path(current_dir) / "state_ablations_config.yaml"
    best_cfg    = Path(current_dir) / "state_ablations_config.yaml"

    if USE_BEST:
        cfg_path = best_cfg
    else:
        cfg_env = os.environ.get("EXPL_CFG", "").strip()
        cfg_path = Path(cfg_env) if cfg_env else default_cfg

    cfg = load_cfg(cfg_path)
    info_msg(f"[runner] using config: {cfg_path}")
    info_msg(f"[runner] USE_BEST={USE_BEST}, DYN_NAME={DYN_NAME}")

    # --- optional env overrides to be exploited by cluster_submit ---
    def _deep_update(dst: dict, src: dict) -> dict:
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    _expl_overrides = os.environ.get("EXPL_OVERRIDES", "").strip()
    if _expl_overrides:
        try:
            ov = json.loads(_expl_overrides)
            before = json.dumps(cfg, sort_keys=True)
            _deep_update(cfg, ov)
            after = json.dumps(cfg, sort_keys=True)
            print("[runner] merged EXPL_OVERRIDES into cfg")
            if before != after:
                print("[runner] overrides diff applied (keys present in EXPL_OVERRIDES took precedence)")
        except Exception as e:
            print(f"[runner] WARNING: failed to parse EXPL_OVERRIDES: {e}")

    # path setup
    run_root = resolve_run_root(current_dir)
    base_out = run_root / "state_space_exploration"
    base_out.mkdir(parents=True, exist_ok=True)

    print("[runner.paths]")
    print(f"  mode:     {'cluster' if (os.environ.get('SUBMITIT_FOLDER') or os.environ.get('SLURM_JOB_ID')) else 'local'}")
    print(f"  run_root: {run_root}")
    print(f"  base_out: {base_out}")

    # ---- seeds & base pools ----
    seeds: List[int]           = list(cfg.get("seeds", [0]))
    train_pools_cfg: Dict[str, Dict[str, List[float]]] = dict(cfg["train_pools"])

    # Legacy fallback if some per-condition pool lists are missing
    fallback_pools: List[str] = list(cfg.get("pools", train_pools_cfg.keys()))

    # Per-condition pool lists (new)
    pool_lists_by_condition: Dict[str, List[str]] = {
        "static_sequential":        list(cfg.get("static_seq_pools",        fallback_pools)),
        "dynamic_sequential":       list(cfg.get("dynamic_seq_pools",       fallback_pools)),
        "static_parallel":          list(cfg.get("static_parallel_pools",   fallback_pools)),
        "dynamic_parallel":         list(cfg.get("dynamic_parallel_pools",  fallback_pools)),
        "extra_static_sequential":  list(cfg.get("extra_static_seq_pools",  [k for k in train_pools_cfg if k.startswith("extra_")] or fallback_pools)),
        "extra_static_parallel":           list(cfg.get("extra_static_parallel_pools",    [k for k in train_pools_cfg if k.startswith("extra_")] or fallback_pools)),
    }

    print("\n[runner.plan] Pools per condition:")
    for _cond, _pools in pool_lists_by_condition.items():
        print(f"  - { _cond }: { _pools }")
    print()

    # Basic PPO settings
    steps_total: int           = int(cfg.get("total_steps", 100_000))
    n_envs_parallel: int       = int(cfg.get("n_envs_parallel", 8))
    # "ppo_n_steps_total" controls the total steps collected per update across all envs.
    # We keep your existing behavior: split it by number of vec envs.
    ppo_n_steps_total: int     = int(cfg.get("ppo_n_steps_total", 2048))

    # Which conditions to run
    conditions: List[str]      = list(cfg.get("conditions", ["static_sequential", "dynamic_sequential"]))

    # ---- evaluation params ----
    eval_cfg = cfg.get("eval", {})
    eval_id_vals      = list(map(float, eval_cfg.get("id",       [0.55])))
    eval_ood_low_vals = list(map(float, eval_cfg.get("ood_low",  [0.35, 0.41])))
    eval_ood_hi_vals  = list(map(float, eval_cfg.get("ood_high", [0.68, 0.75])))
    eval_n_envs       = int(eval_cfg.get("n_envs", 4))
    eval_freq         = int(eval_cfg.get("freq", 10_000))
    eval_eps_per_ctx  = int(eval_cfg.get("episodes_per_ctx", 30))

    # ---- helper to select pools per condition, with validation ----
    def _pools_for_condition(cond: str) -> List[str]:
        if cond not in pool_lists_by_condition:
            raise ValueError(f"Unknown condition '{cond}'. Known: {list(pool_lists_by_condition.keys())}")
        selected = list(pool_lists_by_condition[cond])
        if not selected:
            raise ValueError(f"No pools configured for condition '{cond}'. Please set the corresponding '*_pools' list in YAML.")
        # Validate existence & shape
        bad = [p for p in selected if p not in train_pools_cfg or "length" not in train_pools_cfg[p] or not train_pools_cfg[p]["length"]]
        if bad:
            raise ValueError(f"The following pools are missing or invalid for condition '{cond}': {bad}")
        return selected

    # --- main loops ---
    for seed in seeds:
        for cond in conditions:
            pools_to_run_for_cond = _pools_for_condition(cond)

            runs_spec: List[Tuple[str, List[float]]] = []
            for pool_name in pools_to_run_for_cond:
                vals = train_pools_cfg[pool_name]["length"]
                runs_spec.append((pool_name, [float(v) for v in vals]))

            for pool_name, pool_vals in runs_spec:
                dyn_name_for_path = None  # default

                if "dynamic" in cond:
                    # Look up full spec in either YAML.dynamic or YAML.dynamic_best
                    spec = _dynamic_spec_for_pool(cfg, pool_name, pool_vals)
                    dyn_fn = build_update_fn_from_spec(spec, min_value=LEN_MIN, max_value=LEN_MAX, attribute=CTX_KEY)
                    fns_for_cond = {"length": dyn_fn}

                    # Use the spec name when present, fall back to env DYN_NAME, then a generic tag
                    dyn_name_for_path = spec.get("name") or DYN_NAME or "dynamic"

                    fixed = spec.get("fixed", {}) or {}
                    frac = float(fixed.get("amplitude_frac", 0.0))
                    amp_for_manifest = frac * (LEN_MAX - LEN_MIN) / 2.0
                    period_for_manifest = int(fixed.get("period", 0))
                else:
                    # STATIC: no dynamics, no preset lookup
                    fns_for_cond = {}
                    amp_for_manifest = 0.0
                    period_for_manifest = 0

                # --- OUTPUT DIRECTORY: include dyn name for dynamic runs ---
                if dyn_name_for_path is not None:
                    # e.g. .../pool7/best_continuous_incrementer_pool7/dynamic_sequential/seed_0
                    out = base_out / pool_name / dyn_name_for_path / cond / f"seed_{seed}"
                else:
                    # static / non-dynamic
                    out = base_out / pool_name / cond / f"seed_{seed}"

                (out / "_logs").mkdir(parents=True, exist_ok=True)

                seed_everything(seed)

                vec, manifest = build_vec_for_condition(
                    cond,
                    pool_vals=pool_vals,
                    seed=seed,
                    out_root=out,
                    n_envs_parallel=n_envs_parallel,
                    feature_update_fns=fns_for_cond,
                    parallel_ctx_perm="rotate",
                )

                # per-update steps per env
                steps_per_env = max(1, ppo_n_steps_total // max(1, manifest["n_envs"]))
                vec = GlobalTickTracker(vec, log_csv_path=out / "_logs" / "global_tick.csv")

                manifest.update(dict(
                    seed=seed,
                    total_steps=steps_total,
                    ppo_n_steps_per_env=steps_per_env,
                    ppo_total_steps_target=ppo_n_steps_total,
                    pool=pool_name,
                    sinusoid=dict(
                        amplitude=amp_for_manifest,
                        period=period_for_manifest,
                        len_min=LEN_MIN,
                        len_max=LEN_MAX
                    ),
                ))

                # extra metadata about the pool shape
                manifest.update(dict(
                    pool_size=len(pool_vals),
                    is_dynamic=("dynamic" in cond),
                    dynamic_preset=("large" if len(pool_vals)==1 else "medium" if len(pool_vals)==3 else "small" if len(pool_vals)==7 else None),
                ))

                (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

                # --- build eval contexts for this pool (deterministic) ---
                eval_ctxs_id   = {f"id_{v:.3f}":   {CTX_KEY: v} for v in eval_id_vals}
                eval_ctxs_oodl = {f"oodl_{v:.3f}": {CTX_KEY: v} for v in eval_ood_low_vals}
                eval_ctxs_oodh = {f"oodh_{v:.3f}": {CTX_KEY: v} for v in eval_ood_hi_vals}

                _make_env_for_ctx = make_env_for_ctx_factory(
                    eval_n_envs=eval_n_envs,
                    out_root=out,
                    base_seed=seed * 10_000 + 123,  # fixed source of determinism across reruns
                )

                cb_id = PerContextEvalCallback(
                    ctx_key=CTX_KEY,
                    make_env_for_ctx=_make_env_for_ctx,
                    eval_ctxs=eval_ctxs_id,
                    eval_freq=eval_freq,
                    n_episodes_per_ctx=eval_eps_per_ctx,
                    deterministic=True,
                    seed_summary_dir=out / "_eval",
                    tag="id",
                    eval_n_envs=eval_n_envs,
                    verbose=0,
                )
                cb_oodl = PerContextEvalCallback(
                    ctx_key=CTX_KEY,
                    make_env_for_ctx=_make_env_for_ctx,
                    eval_ctxs=eval_ctxs_oodl,
                    eval_freq=eval_freq,
                    n_episodes_per_ctx=eval_eps_per_ctx,
                    deterministic=True,
                    seed_summary_dir=out / "_eval",
                    tag="ood_low",
                    eval_n_envs=eval_n_envs,
                    verbose=0,
                )
                cb_oodh = PerContextEvalCallback(
                    ctx_key=CTX_KEY,
                    make_env_for_ctx=_make_env_for_ctx,
                    eval_ctxs=eval_ctxs_oodh,
                    eval_freq=eval_freq,
                    n_episodes_per_ctx=eval_eps_per_ctx,
                    deterministic=True,
                    seed_summary_dir=out / "_eval",
                    tag="ood_high",
                    eval_n_envs=eval_n_envs,
                    verbose=0,
                )
                callbacks = CallbackList([cb_id, cb_oodl, cb_oodh])

                model = PPO(
                    "MlpPolicy",
                    vec,
                    verbose=1,
                    n_steps=steps_per_env,
                    tensorboard_log=str(out / "tb")
                )
                model.learn(total_timesteps=steps_total,
                            tb_log_name=f"ppo_{pool_name}_{cond}_seed{seed}",
                            callback=callbacks)
                model.save(str(out / "ppo_final.zip"))

                try:
                    mean_r, std_r = evaluate_policy(model, vec, n_eval_episodes=10, deterministic=True)
                    print(f"[{pool_name} | {cond} | seed {seed}] meanR={mean_r:.2f} ± {std_r:.2f}")
                except Exception as e:
                    print(f"Eval failed ({pool_name} | {cond} | seed {seed}): {e}")

                vec.close()
