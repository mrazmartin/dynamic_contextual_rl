#!/usr/bin/env python3
from __future__ import annotations

import os, sys
from pathlib import Path
from typing import Tuple

from torch import nn  # add this import at the top

# --- keep project root on sys.path (mirrors your style) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../../../..")
sys.path.append(project_root)

# --- RL imports ---
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# --- your helpers ---
from dynamic_crl.helix_utils.helix_submitit import run_with_submitit
from dynamic_crl.src.dmc_envs.loader import load_dmc_env
from dynamic_crl.src.dmc_envs.mj_gym_wrapper import MujocoToGymWrapper

# start TB logging of our custom metrics
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Callable
from stable_baselines3.common.evaluation import evaluate_policy

class EvalInfoLogger(BaseCallback):
    """
    Every `every_steps`, build a fresh eval env via `make_eval_env()`,
    run `n_episodes` deterministic rollouts, collect info['eval_log'] at LAST,
    and log eval_v2/* scalars to TensorBoard. Does NOT touch SB3's EvalCallback.
    """
    def __init__(self, make_eval_env: Callable[[], Any], every_steps: int = 20000, n_episodes: int = 10):
        super().__init__()
        self.make_eval_env = make_eval_env
        self.every_steps = int(every_steps)
        self.n_episodes = int(n_episodes)

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.every_steps != 0:
            return True

        # fresh eval env each time, so we don't interfere with SB3's EvalCallback
        eval_env = self.make_eval_env()

        # rollouts
        per_ep: List[Dict[str, Any]] = []

        for _ in range(self.n_episodes):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)
                if done and isinstance(info, dict) and ("eval_log" in info):
                    per_ep.append(dict(info["eval_log"]))

        # aggregate
        def _mean(key):
            vals = [ep.get(key) for ep in per_ep if ep.get(key) is not None]
            return float(np.mean(vals)) if vals else np.nan

        success_rate = _mean("success")          # 0/1 per episode
        best_h_mean  = _mean("best_height")
        tts_mean     = _mean("time_to_stand_s")
        # and the means
        avg_h_mean   = _mean("avg_height")
        avg_tau_rms  = _mean("avg_action_rms")

        # log + flush
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/best_height_mean", best_h_mean)
        self.logger.record("eval/time_to_stand_s_mean", tts_mean)
        self.logger.record("eval/avg_height_mean", avg_h_mean)
        self.logger.record("eval/avg_action_rms", avg_tau_rms)
        self.logger.dump(self.model.num_timesteps)

        # clean up
        try:
            eval_env.close()
        except Exception:
            pass

        return True
# end TaskMetricsCallback

# =========================
# Env factory (single env)
# =========================
def make_gym_env(
    *,
    seed: int = 0,
    start_key: str = "sit",
    min_alive_height: float = 0.05,
    max_tilt_deg: float = 60.0,
    time_limit_s: float = 20.0,
    visualize_reward: bool = False,
    eval_mode: bool = False,
    # ---- control selection ----
    control_type: str = "torque",   # or "pd"
    stiffness: float = 20.0,        # Kp when PD
    damping: float = 0.5,           # Kd when PD
    action_scale: float = 0.25,      # rad per unit action (PD residual)
):
    """Create dm_control StandUp env and wrap in Gymnasium API."""
    task_kwargs = dict(
        start_key=start_key,
        min_alive_height=min_alive_height,
        max_tilt_deg=max_tilt_deg,
        # PD control params
        control_type=control_type,
        stiffness=stiffness,
        damping=damping,
        action_scale=action_scale,
    )

    environment_kwargs = dict(time_limit=time_limit_s)  # loader sets flat_observation=True by default

    dm_env = load_dmc_env(
        domain_name="go2",
        task_name="stand_up",  # must exist in dmc_go2.SUITE
        task_kwargs=task_kwargs, # physics/task parameters
        environment_kwargs=environment_kwargs,
        visualize_reward=visualize_reward,
    )
    # IMPORTANT: RESET THE ENV BEFORE WRAPPING TO INITIALIZE IT! (START POSE/PHYSICS etc)
    dm_env.reset()  # this will initialize the physics/model
    # wrap to Gym after correct pose init
    gym_env = MujocoToGymWrapper(dm_env, eval_mode=eval_mode)
    gym_env.reset(seed=seed)
    return Monitor(gym_env)


# =========================
# Main training (runs inside allocation)
# =========================
def ppo_train_standup(results_root: Path) -> Tuple[float, float]:
    """
    Train PPO to stand up from 'sit' using our DMC->Gym wrapper.
    Runs **once** (no sweeps), returns (mean_eval_return, std_eval_return).
    """

    version = "test_paw_reward"

    # --------- fixed settings you can edit here ---------
    SEED = 1
    TIMESTEPS = 5_000_000

    # --- fixed settings ---
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(SEED)
    except Exception:
        pass

    # PPO hyperparams
    DEVICE = "cpu"  # PPO+MLP is usually faster on CPU

    # Eval/checkpoints
    EVAL_EPISODES = 10
    EVAL_FREQ = 20_000
    SAVE_FREQ = 250_000

    # control HPs
    control_type = "pd"  # or "torque"
    stiffness = 40.0     # Kp when PD
    damping = 1.0        # Kd when PD
    action_scale = 0.50   # rad per unit action (PD residual)

    # env params
    time_limit_train = 10.0  # seconds
    # ----------------------------------------------------

    # Directory layout inside results_root
    logdir = results_root / "go2_results" / f"ppo_go2_standup_{version}"
    (logdir / "best").mkdir(parents=True, exist_ok=True)
    (logdir / "ckpts").mkdir(parents=True, exist_ok=True)

    # Auto-enable TensorBoard if present
    try:
        import tensorboard as _tb  # noqa: F401
        tb_dir = str(logdir)
    except Exception:
        tb_dir = None

    # ---------- Build envs ----------
    train_thunk = lambda: make_gym_env(
        seed=SEED,
        start_key="sit",
        min_alive_height=0.05,
        max_tilt_deg=60.0,
        time_limit_s=time_limit_train,
        # control
        control_type=control_type,
        stiffness=stiffness,
        damping=damping,
        action_scale=action_scale,
        )
    vec_env = DummyVecEnv([train_thunk])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        )

    eval_thunk = lambda: make_gym_env(
        seed=10_000,
        start_key="sit",
        min_alive_height=0.05,
        max_tilt_deg=60.0,
        time_limit_s=8.0,
        control_type=control_type,
        stiffness=stiffness,
        damping=damping,
        action_scale=action_scale,
        eval_mode=True,
        )
    eval_env = DummyVecEnv([eval_thunk])
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        )
    eval_env.obs_rms = vec_env.obs_rms  # sync stats

    # ---------- Model ----------
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,        # was 2e-4
        n_steps=2048,              # was 2048
        batch_size=256,            # 1024 rollout / 256 = 4 minibatches
        n_epochs=10,                # was 10
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.045,            # was 0.03
        ent_coef=5e-3,             # was 1e-3
        vf_coef=0.7,               # was 0.7
        max_grad_norm=0.5,         # was 0.5

        # --- SDE/exploration: put use_sde here (top-level) ---
        use_sde=True,
        sde_sample_freq=4,         # resample exploration noise every 4 env steps

        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ELU,  # requires torch.nn as nn
            log_std_init=0.6,      # higher initial exploration
            ortho_init=False,      # ELU + larger nets: leave off
        ),

        tensorboard_log=tb_dir,
        seed=SEED,
        verbose=1,
        device=DEVICE,
    )

    # ---------- Callbacks ----------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(logdir / "best"),
        log_path=str(logdir / "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=str(logdir / "ckpts"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    # ---------- Train ----------
    eval_cb_std = EvalInfoLogger(
        make_eval_env=eval_thunk,
        every_steps=EVAL_FREQ,
        n_episodes=EVAL_EPISODES
        )
    model.learn(total_timesteps=TIMESTEPS, callback=[eval_cb, ckpt_cb, eval_cb_std])

    # ---------- Save artifacts ----------
    model.save(str(logdir / "final_model"))
    vec_env.save(str(logdir / "vecnorm.pkl"))

    # ---------- Final eval ----------
    eval_env.obs_rms = vec_env.obs_rms
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    print(f"[FINAL EVAL] mean={mean_r:.3f} ± {std_r:.3f}")

    vec_env.close()
    eval_env.close()
    return float(mean_r), float(std_r)


# =========================
# Submitit entrypoints
# =========================
def ppo_main_wrapped(results_root: str):
    """Submitit calls this inside the allocation."""
    results_root_path = Path(results_root)
    results_root_path = Path("/home/martin/Code/Thesis/dynamic_crl/src/dmc_envs/dc_envs/test_scripts/run_logs/stand_up")
    results_root_path.mkdir(parents=True, exist_ok=True)
    return ppo_train_standup(results_root_path)


if __name__ == "__main__":
    # Submit a single PPO job to your cluster.
    # Adjust minutes/mem if needed; PPO here is CPU-only by default.

    version_suffix = "test_paw_reward"  # also edit inside ppo_train_standup()

    run_with_submitit(
        main_fn=ppo_main_wrapped,
        submit=None,                  # let your helper decide via env var
        ws_name="c_rl-exp",
        cpu=True,                     # CPU partition
        minutes=1200,                 # 20 hours (edit as you like)
        cpus=2,                       # 1–2 is fine
        mem_gb=8,                     # enough for Mujoco + SB3
        gpu_type=None,
        part_cpu="cpu-single",
        part_gpu="gpu-single",
        link_target_subdir="runs",
        run_name=f"ppo_go2_standup_{version_suffix}",
    )
