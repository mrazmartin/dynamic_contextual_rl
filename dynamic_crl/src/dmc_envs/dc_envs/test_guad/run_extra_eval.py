#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Any, Dict, Callable

import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# --- project path (adjust if needed) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../../../..")
sys.path.append(project_root)

# --- your env ---
from dynamic_crl.src.dmc_envs.dc_envs.carl_dm_quadruped import CARLDmcQuadrupedEnv as dQuadEnv

# ===== config: MUST match training =====
OBS_CONTEXT_FEATURES = ["gravity"]

TRAIN_CONTEXTS: Dict[int, Dict[str, float]] = {
    0: {'gravity': 9.8},
    1: {'gravity': 9.8},
}
EVAL_CONTEXTS: Dict[int, Dict[str, float]] = {
    0: {'gravity': 9.8},
    1: {'gravity': 9.8},
}

DC_UPDATERS: Dict[str, Any] = {}
EASY_INIT = True

# paths
VERSION = "easy_init"
STEPS = 5_000_000  # adjust if you want a checkpoint instead of final
RUN_DIR = Path("/home/martin/Code/Thesis/dynamic_crl/src/dmc_envs/dc_envs/test_carl/run_logs/quadraped") / f"ppo_quad_run_{VERSION}"

MODEL_PATH   = RUN_DIR / "final_model.zip"             # from model.save("final_model")
VECNORM_PATH = RUN_DIR / "vecnorm.pkl"                 # from vec_env.save("vecnorm.pkl")
# If you want a checkpoint instead, use:
# MODEL_PATH   = RUN_DIR / "ckpts" / f"ppo_{STEPS}_steps.zip"
# VECNORM_PATH = RUN_DIR / "ckpts" / f"ppo_vecnormalize_{STEPS}_steps.pkl"

EVAL_EPISODES = 10
DEVICE = "cpu"  # keep CPU (your PPO is MLP)

def make_train_env():
    return Monitor(FlattenObservation(
        dQuadEnv(
            render_mode=None,
            contexts=TRAIN_CONTEXTS,
            dc_updaters=DC_UPDATERS,
            quad_env_kwargs={"scale_toe_fric": False},
            environment_kwargs={"flat_observation": True},
            task_kwargs={"time_limit": 6.0, "random": 10, "easy_init": EASY_INIT},
            obs_context_features=OBS_CONTEXT_FEATURES,
        )
    ))

def make_eval_env():
    return Monitor(FlattenObservation(
        dQuadEnv(
            render_mode=None,
            contexts=EVAL_CONTEXTS,
            dc_updaters=DC_UPDATERS,
            quad_env_kwargs={"scale_toe_fric": False},
            environment_kwargs={"flat_observation": True},
            task_kwargs={"time_limit": 10.0, "random": 1000, "is_evaluation": True, "easy_init": EASY_INIT},
            obs_context_features=OBS_CONTEXT_FEATURES,
        )
    ))


def main():
    assert MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}"
    assert VECNORM_PATH.exists(), f"Missing VecNormalize stats at {VECNORM_PATH}"

    # ---------- Rebuild eval vecenv & attach VecNormalize ----------
    eval_vecenv = DummyVecEnv([make_eval_env])
    eval_vecnorm: VecNormalize = VecNormalize.load(str(VECNORM_PATH), eval_vecenv)
    eval_vecnorm.training = False
    eval_vecnorm.norm_reward = True  # keep as in training; we'll also compute RAW below

    # Sanity: obs dim must match stats
    flat_dim_now = int(eval_vecnorm.observation_space.shape[0])
    flat_dim_stats = int(eval_vecnorm.obs_rms.mean.shape[0])
    if flat_dim_now != flat_dim_stats:
        raise RuntimeError(
            f"Obs dim mismatch: env={flat_dim_now} vs vecnorm_stats={flat_dim_stats}. "
            f"Check obs construction / OBS_CONTEXT_FEATURES order."
        )

    # ---------- Load model on CPU and bind env ----------
    model: PPO = PPO.load(str(MODEL_PATH), device=DEVICE)
    model.set_env(eval_vecnorm)

    # ---------- RAW evaluation (turn off reward normalization) ----------
    eval_vecnorm_raw = VecNormalize.load(str(VECNORM_PATH), DummyVecEnv([make_eval_env]))
    eval_vecnorm_raw.training = False
    eval_vecnorm_raw.norm_reward = False  # raw task rewards
    mean_r_raw, std_r_raw = evaluate_policy(
        model, eval_vecnorm_raw, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )
    print(f"[EVAL RAW]  episodes={EVAL_EPISODES}  mean_return={mean_r_raw:.3f}  std={std_r_raw:.3f}")
    eval_vecnorm_raw.close()

    # ---------- NORMALIZED evaluation (manual rollout) ----------
    # evaluate_policy may still report raw totals depending on wrappers;
    # do a manual loop to sum the *normalized* rewards explicitly.
    eval_vecnorm.training = False
    eval_vecnorm.norm_reward = True

    obs = eval_vecnorm.reset()
    norm_returns, raw_returns = [], []
    ep_norm, ep_raw = 0.0, 0.0

    while len(raw_returns) < EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_vecnorm.step(action)  # reward is normalized (since norm_reward=True)

        # reward: normalized
        ep_norm += float(reward[0])

        # denormalized: use VecNormalize.get_original_reward() with NO arguments
        raw_step = eval_vecnorm.get_original_reward()  # returns array for all envs from last step
        ep_raw += float(raw_step[0])

        if done[0]:
            norm_returns.append(ep_norm)
            raw_returns.append(ep_raw)
            ep_norm, ep_raw = 0.0, 0.0
            obs = eval_vecnorm.reset()

    print(f"[ROLLOUT NORM] mean={np.mean(norm_returns):.3f}  std={np.std(norm_returns):.3f}  all={np.round(norm_returns,3)}")
    print(f"[ROLLOUT RAW ] mean={np.mean(raw_returns):.3f}  std={np.std(raw_returns):.3f}  all={np.round(raw_returns,3)}")

    eval_vecnorm.close()

if __name__ == "__main__":
    # Make runs deterministic like your training script did (optional)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
