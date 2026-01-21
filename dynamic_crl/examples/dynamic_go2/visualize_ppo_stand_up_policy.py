#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

# --- project path (adjust if needed) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../../../..")
sys.path.append(project_root)

# --- your task/env ---
from dynamic_crl.src.dmc_envs.dc_envs.dmc_tasks import dmc_go2
from dm_control import viewer

from dynamic_crl.src.dmc_envs.mj_gym_wrapper import MujocoToGymWrapper

# --- SB3 + VecNormalize ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces

# ======== Playback knobs (tune here) ========
DETERMINISTIC     = False     # allow stochastic sampling
ACTION_GAIN       = 1       # 1.0–3.0; multiplies actions before clamp
ACTION_NOISE_STD  = 0.0      # 0.0 to disable
EMA_ALPHA         = 0.10      # 0.0 disables smoothing; 0.1–0.3 smooths
REPEAT            = 1         # repeat last action N steps (1 = off)
# ===========================================

def _build_vecnorm_normalizer(vecnorm_path: str | None, obs_space: spaces.Box):
    if not vecnorm_path or not Path(vecnorm_path).exists():
        return None

    # Minimal dummy env so VecNormalize can load and normalize obs for us
    import gymnasium as gym
    class _DummyObsEnv(gym.Env):
        def __init__(self, obs_space, act_space):
            self.observation_space = obs_space
            self.action_space = act_space
        def reset(self, *, seed=None, options=None):
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}
        def step(self, action):
            return (np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype),
                    0.0, False, False, {})

    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    dummy_vec = DummyVecEnv([lambda: _DummyObsEnv(obs_space, act_space)])

    vecnorm = VecNormalize.load(vecnorm_path, dummy_vec)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm

def make_sb3_policy(model: PPO, vecnorm: VecNormalize | None, env):
    """
    Returns a dm_control viewer policy: TimeStep -> action (np.ndarray).
    Applies VecNormalize obs normalization if provided, adds test-time knobs.
    """
    # cache actuator limits from the real env
    a_spec = env.action_spec()
    low  = np.asarray(a_spec.minimum, dtype=np.float32).reshape(-1)
    high = np.asarray(a_spec.maximum, dtype=np.float32).reshape(-1)

    # state for smoothing/repeat/warmup
    step = 0
    last_action = np.zeros_like(low, dtype=np.float32)
    repeat_left = 0

    def policy(timestep):
        nonlocal step, last_action, repeat_left

        # 1) get obs vector
        obs = timestep.observation
        if isinstance(obs, dict):
            obs = obs.get("observations", None)
        if obs is None:
            raise RuntimeError("TimeStep.observation does not contain 'observations' key.")
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)  # (1, obs_dim)

        # 2) normalize (same stats as training)
        if vecnorm is not None:
            obs = vecnorm.normalize_obs(obs)

        # 3) choose action (repeat -> model)
        if repeat_left > 0:
            a = last_action
            repeat_left -= 1
        else:
            a, _ = model.predict(obs, deterministic=DETERMINISTIC)
            a = np.asarray(a, dtype=np.float32).reshape(-1)

            # noise at test time
            if ACTION_NOISE_STD > 0:
                a = a + np.random.normal(0.0, ACTION_NOISE_STD, size=a.shape).astype(np.float32)

            # gain + smoothing + clamp
            a = a * ACTION_GAIN
            if EMA_ALPHA > 0.0:
                a = EMA_ALPHA * last_action + (1.0 - EMA_ALPHA) * a
            a = np.clip(a, low, high)

            last_action = a
            print("- - - - - - - - - - - - - - - - -")
            print(f"agent acts: {last_action}")
            repeat_left = max(1, REPEAT) - 1

        step += 1
        return a

    return policy

def main():
    # Use the same factory used for training!!! use the same kwargs too!!!
    env = dmc_go2.stand_up(
        control_type="pd",  # torque or pd for joint actuation
        # PD control
        stiffness=20.0,     # Kp when PD
        damping=0.5,        # Kd when PD
        action_scale=0.5,  # rad per unit action (PD residual)

        # for our visualization we can edit the following as we want
        environment_kwargs=dict(time_limit=float("inf")),
        start_key="sit", min_alive_height=0.05, max_tilt_deg=60.0,
    )

    # Load SB3 model
    model = PPO.load(MODEL_PATH, device="cpu")

    # Load VecNormalize stats
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
    vecnorm = _build_vecnorm_normalizer(VECNORM_PATH, obs_space)

    # make debug prints visible
    env.physics._debug_prints = True

    # Viewer policy
    policy = make_sb3_policy(model, vecnorm, env)

    # Sanity: PD mode + PD action bounds [-1,1]
    a_spec = env.action_spec()
    print("control_type:", env.physics.control_type)
    print("action_spec min/max:", np.min(a_spec.minimum), np.max(a_spec.maximum))

    viewer.launch(env, policy=policy)
    return env, policy

if __name__ == "__main__":
    # os.environ.setdefault("MUJOCO_GL", "egl")  # for headless/EGL setups

    version = "test_paw_reward"  # also edit above
    step = int(250_000)  # edit to load different checkpoint

    MODEL_PATH   = f"/home/martin/Code/Thesis/dynamic_crl/src/dmc_envs/dc_envs/test_scripts/run_logs/go2_results/ppo_go2_standup_{version}/ckpts/ppo_{step}_steps.zip"
    VECNORM_PATH = f"/home/martin/Code/Thesis/dynamic_crl/src/dmc_envs/dc_envs/test_scripts/run_logs/go2_results/ppo_go2_standup_{version}/ckpts/ppo_vecnormalize_{step}_steps.pkl"
    env, policy = main()

    save_gif = False

    if save_gif:
        from dynamic_crl.src.dmc_envs.dmc_utils.saving_mj_run_gifs import record_gif_from_env
        from dynamic_crl.src.dmc_envs.dc_envs.go2_as_dmc import CARLDmcGo2Env as Go2Env
        # make the env mujoco-gym wrapped so we can use our GIF recorder utility
        def make_ctx_go2_env(**kwargs):
            return Go2Env(**kwargs)
        
        env = make_ctx_go2_env(render_mode="rgb_array", contexts={0: {"gravity_z": -9.8, "payload_mass": 1.0}}, dc_updaters={})

        out_gif_path = f"/home/martin/Code/Thesis/dynamic_crl/src/dmc_envs/dc_envs/test_scripts/run_logs/go2_results/ppo_go2_standup_{version}/videos/ppo_standup_{version}_{step}steps.gif"
        print(f"Recording GIF to: {out_gif_path}")
        record_gif_from_env(
            go2_env=env,
            out_path=out_gif_path,
            steps=1000,
            size=(480, 640),
            camera_id=None,  # None = auto-pick world-fixed if available
            policy_fn=policy,
            slowmo=1.0,
            viewer_min_delay_ms=40,  # adjust if your viewer is faster/slower
            stop_on_first_episode=True,
            include_terminal_frame=True,
        )
