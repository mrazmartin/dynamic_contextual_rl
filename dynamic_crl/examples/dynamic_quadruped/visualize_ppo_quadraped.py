#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Any, Dict

# TODO: this is just a static policy visualization

import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from collections import OrderedDict

# --- project path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../../../..")
sys.path.append(project_root)

from dynamic_crl.src.dmc_envs.dc_envs.carl_dm_quadruped import CARLDmcQuadrupedEnv as dQuadEnv

# ===== config =====
OBS_CONTEXT_FEATURES = ["gravity"]

# THE ORDER OF CONTEXTS HERE MUST MATCH THE ORDER IN TRAINING DICTIONARIES
EVAL_CONTEXTS  = {0: {'gravity': 9.8}, 1: {'gravity': 9.8}}

DC_UPDATERS: Dict[str, Any] = {}
EASY_INIT = True

# paths
VERSION = "gravity_triplet_ctx"
TIMESTEP = 1_500_000
RUN_DIR = Path("/home/martin/Code/Thesis/dynamic_crl/src/dmc_envs/dc_envs/test_carl/run_logs/quadraped") / f"ppo_quad_run_{VERSION}"
MODEL_PATH   = RUN_DIR / "final_model.zip" if TIMESTEP is None else RUN_DIR / f"ckpts/ppo_{TIMESTEP}_steps.zip"
VECNORM_PATH = RUN_DIR / "vecnorm.pkl" if TIMESTEP is None else RUN_DIR / f"ckpts/ppo_vecnormalize_{TIMESTEP}_steps.pkl"
EVAL_EPISODES = 10
DEVICE = "cpu"

# === env builders ===
def make_env(contexts):
    return Monitor(FlattenObservation(
        dQuadEnv(
            render_mode=None,
            contexts=contexts,
            dc_updaters=DC_UPDATERS,
            quad_env_kwargs={"scale_toe_fric": False},
            environment_kwargs={"flat_observation": True},
            task_kwargs={"time_limit": 10.0, "random": 1000, "is_evaluation": True, "easy_init": EASY_INIT},
            obs_context_features=OBS_CONTEXT_FEATURES,
        )
    ))

# === manual rollout helper ===
def manual_rollout(model: PPO, vecnorm: VecNormalize, n_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Run manual rollouts and collect both normalized + raw rewards."""
    obs = vecnorm.reset()
    norm_returns, raw_returns = [], []
    ep_norm, ep_raw = 0.0, 0.0

    while len(raw_returns) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vecnorm.step(action)

        ep_norm += float(reward[0])  # normalized reward
        ep_raw  += float(vecnorm.get_original_reward()[0])  # raw reward

        if done[0]:
            norm_returns.append(ep_norm)
            raw_returns.append(ep_raw)
            ep_norm, ep_raw = 0.0, 0.0
            obs = vecnorm.reset()

    return np.array(norm_returns), np.array(raw_returns)

# === view policy in env ===
# === viewer policy that drives dm_control.viewer with the PPO model ===
def make_viewer_policy(model: PPO, vecnorm_stats_path: Path | str | None):
    # Load VecNormalize stats into a dummy VecEnv so we can use normalize_obs()
    vn = None
    if vecnorm_stats_path is not None and Path(vecnorm_stats_path).exists():
        dummy = DummyVecEnv([lambda: make_env(EVAL_CONTEXTS)])
        vn = VecNormalize.load(str(vecnorm_stats_path), dummy)
        vn.training = False
        vn.norm_reward = False
        vn.norm_obs = True

    def _flatten_obs(obs: Any) -> np.ndarray:
        # Handles dict obs (dm_control) or already-flat arrays
        if isinstance(obs, dict):
            # deterministic key order so it matches FlattenObservation
            parts = []
            for k in sorted(obs.keys()):
                v = obs[k]
                v = np.asarray(v)
                parts.append(v.ravel())
            return np.concatenate(parts, dtype=np.float32)
        else:
            arr = np.asarray(obs, dtype=np.float32)
            return arr.ravel()

    def viewer_policy(time_step_or_obs):
        # Some hook envs pass a dm_env.TimeStep, others pass a flat obs already.
        obs = getattr(time_step_or_obs, "observation", time_step_or_obs)
        obs = _flatten_obs(obs).astype(np.float32)

        if vn is not None:
            # VecNormalize expects a batch
            obs = vn.normalize_obs(obs[None, :])[0]

        action, _ = model.predict(obs, deterministic=True)
        # ensure np.ndarray (viewer is fine with list/array)
        return np.asarray(action, dtype=np.float32)

    return viewer_policy

def main(run_manual_rollout: bool = False):
    assert MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}"
    assert VECNORM_PATH.exists(), f"Missing VecNormalize stats at {VECNORM_PATH}"

    # ---- load model ----
    model: PPO = PPO.load(str(MODEL_PATH), device=DEVICE)

    if run_manual_rollout:
        # ---- RAW rollout ----
        raw_env = DummyVecEnv([lambda: make_env(EVAL_CONTEXTS)])
        raw_vecnorm = VecNormalize.load(str(VECNORM_PATH), raw_env)
        raw_vecnorm.training = False
        raw_vecnorm.norm_reward = False
        model.set_env(raw_vecnorm)
        norm_r, raw_r = manual_rollout(model, raw_vecnorm, EVAL_EPISODES)
        print(f"[RAW] mean={np.mean(raw_r):.3f} ±{np.std(raw_r):.3f}  all={np.round(raw_r,3)}")
        raw_vecnorm.close()

        # ---- NORMALIZED rollout ----
        norm_env = DummyVecEnv([lambda: make_env(EVAL_CONTEXTS)])
        norm_vecnorm = VecNormalize.load(str(VECNORM_PATH), norm_env)
        norm_vecnorm.training = False
        norm_vecnorm.norm_reward = True
        model.set_env(norm_vecnorm)
        norm_r2, raw_r2 = manual_rollout(model, norm_vecnorm, EVAL_EPISODES)
        print(f"[NORM] mean={np.mean(norm_r2):.3f} ±{np.std(norm_r2):.3f}  all={np.round(norm_r2,3)}")
        norm_vecnorm.close()

    # ---- VISUALIZATION ----
    # build a single DMC env, wrapped so its flattened
    vis_env = make_env(EVAL_CONTEXTS)
    # get the policy from the model
    vis_env.env.env._viewer_policy = make_viewer_policy(model, VECNORM_PATH)
    # launch the dm_control viewer via your env's render() (it should call viewer.launch(...))
    vis_env.env.env._render_mode = "human"
    vis_env.reset()
    vis_env.render()

# now saving to a gif
def make_env_rgb(contexts):
    # same as make_env but render_mode='rgb_array'
    return Monitor(FlattenObservation(
        dQuadEnv(
            render_mode="rgb_array",
            contexts=contexts,
            dc_updaters=DC_UPDATERS,
            quad_env_kwargs={"scale_toe_fric": False},
            environment_kwargs={"flat_observation": True},
            task_kwargs={"time_limit": 10.0, "random": 1000, "is_evaluation": True, "easy_init": EASY_INIT},
            obs_context_features=OBS_CONTEXT_FEATURES,
        )
    ))

import imageio
def rollout_to_gif(model, vecnorm_path, contexts, out_path, steps=1000, fps=25, frame_skip=2, size=(480, 640)):
    """
    Runs a rollout using SB3 model+VecNormalize and records frames to a GIF.
    - Uses DummyVecEnv([make_env_rgb]) + VecNormalize.load(stats).
    - Grabs frames from the base env (render()) or dm_control physics as fallback.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build vec env with RGB rendering
    venv = DummyVecEnv([lambda: make_env_rgb(contexts)])
    vn = VecNormalize.load(str(vecnorm_path), venv)
    vn.training = False
    vn.norm_reward = False
    vn.norm_obs = True

    model.set_env(vn)

    # base env for rendering (the one inside DummyVecEnv)
    base_env = vn.venv.envs[0]

    # reset
    obs = vn.reset()
    frames = []

    H, W = size
    def grab_frame():
        # Try gym-style render first
        frame = None

        try:
            frame = base_env.render()  # returns RGB array if supported
        except Exception:
            frame = None
        if frame is None:
            # dm_control fallback via physics.render
            physics = getattr(base_env.unwrapped, "physics", None)
            if physics is None and hasattr(base_env.unwrapped, "env"):
                physics = getattr(base_env.unwrapped.env, "physics", None)
            if physics is None:
                raise RuntimeError("Could not locate dm_control physics to render a frame.")
            frame = physics.render(height=H, width=W, camera_id=None)
        return np.asarray(frame)

    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vn.step(action)

        if (t % frame_skip) == 0:
            frames.append(np.asarray(grab_frame()))

        if done[0]:
            # include terminal frame
            frames.append(np.asarray(grab_frame()))
            break

    imageio.mimsave(out_path, frames, duration=1.0 / fps)
    print(f"[gif] wrote {len(frames)} frames to {out_path}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    
    if True:
        main(run_manual_rollout=False)

    if False:
        out_gif_path = RUN_DIR / f"videos/ppo_quad_{VERSION}_{TIMESTEP}steps.gif"
        print(f"Saving GIF to: {out_gif_path}")

        model: PPO = PPO.load(str(MODEL_PATH), device=DEVICE)
        rollout_to_gif(model, VECNORM_PATH, EVAL_CONTEXTS, out_gif_path, steps=1000,
                       fps=25, frame_skip=4, size=(480, 640))
