#!/usr/bin/env python3
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# --- project path (adjust if needed) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../../../..")
sys.path.append(project_root)

# --- your loaders/wrappers ---
from dynamic_crl.src.dmc_envs.mj_gym_wrapper import MujocoToGymWrapper
from dynamic_crl.src.dmc_envs.loader import load_dmc_env

from stable_baselines3.common.monitor import Monitor
# optional: SB3 imports if you want to drive the viewer with a policy
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from gymnasium import spaces
from dm_control import viewer


def make_envs(
    time_limit_s: float = 10.0,
    eval_mode: bool = False,
    visualize_reward: bool = False,
    seed: int = 0,
):
    """
    Returns (gym_env, dm_env).
    gym_env is a Monitor-wrapped MujocoToGymWrapper around dm_env (for matplotlib).
    dm_env is the raw dm_control env (for viewer).
    """
    environment_kwargs = dict(time_limit=time_limit_s)
    task_kwargs = dict()

    # IMPORTANT: this must hit your custom suite where 'stay_still' is registered
    dm_env = load_dmc_env(
        domain_name="quadruped",
        task_name="stay_still_context",  # or "stay_still_context" if that's what you added
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        visualize_reward=visualize_reward,
    )

    # wrap for gym-style render/step loop (matplotlib path)
    dm_env.reset()
    gym_env = MujocoToGymWrapper(dm_env, eval_mode=eval_mode)
    gym_env.reset(seed=seed)

    # convenience: expose dm_env back through the wrapper
    # (so you can retrieve it later as gym_env.unwrapped_dm)
    try:
        gym_env.unwrapped_dm = dm_env
    except Exception:
        pass

    return Monitor(gym_env), dm_env

def build_env(seed: int = 0):
    return load_dmc_env(
        domain_name="quadruped",
        task_name="stay_still_context",
        task_kwargs=dict(random=seed),
        environment_kwargs=dict(time_limit=10.0),
        visualize_reward=False,
    )

# -------- viewer policies --------
def random_policy_for(dm_env):
    """Simple random policy in dm_control action_spec bounds."""
    a_spec = dm_env.action_spec()
    low = np.asarray(a_spec.minimum, dtype=np.float32).reshape(-1)
    high = np.asarray(a_spec.maximum, dtype=np.float32).reshape(-1)

    def _policy(timestep):
        return np.random.uniform(low, high).astype(np.float32)
    return _policy
# ---------------------------------
# -------- trained policy --------
# def load_sb3_policy(model_path: str | Path, vecnorm_path: str | None, dm_env):
#     """
#     Returns a dm_control viewer policy: TimeStep -> action (np.ndarray).
#     Applies VecNormalize obs normalization if provided.
#     """
#     if not Path(model_path).exists():
#         raise FileNotFoundError(f"Model path {model_path} does not exist.")
#     model = PPO.load(model_path)
#     vecnorm = _build_vecnorm_normalizer(vecnorm_path, dm_env.observation_spec())
#     return make_sb3_policy(model, vecnorm, dm_env)
# ---------------------------------

if __name__ == "__main__":
    # ---- choose which path to test ----
    USE_VIEWER = True   # True => dm_control viewer; False => matplotlib loop

    # For EGL (headless or NVIDIA):
    # os.environ.setdefault("MUJOCO_GL", "egl")

    gym_env, dm_env = make_envs(time_limit_s=10.0, eval_mode=False, visualize_reward=False, seed=0)

    if USE_VIEWER:
        # Pick a policy for the viewer
        # policy = random_policy_for(dm_env)
        policy = random_policy_for(dm_env)   # good for stay_still

        # nice diagnostics
        a_spec = dm_env.action_spec()
        print("Action spec min/max:", np.min(a_spec.minimum), np.max(a_spec.maximum))

        build_env_lmb = lambda: build_env(seed=3)

        viewer.launch(environment_loader=build_env_lmb, policy=random_policy_for(dm_env))


    else:
        # Matplotlib rendering path using the Gym wrapper (rgb_array frames)
        obs, info = gym_env.reset()
        for _ in range(300):
            # MujocoToGymWrapper should support rgb_array mode
            frame = gym_env.env.render(mode="rgb_array")
            if frame is not None:
                plt.imshow(frame)
                plt.axis("off")
                plt.pause(0.001)

            action = gym_env.action_space.sample()
            obs, reward, terminated, truncated, info = gym_env.step(action)
            if terminated or truncated:
                obs, info = gym_env.reset()
