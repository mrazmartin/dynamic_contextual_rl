#!/usr/bin/env python3
from __future__ import annotations

import os, sys
from pathlib import Path
from typing import Tuple, Any, Dict, List, Callable

from torch import nn

# --- keep project root on sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../../../..")
sys.path.append(project_root)

# --- RL imports ---
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# --- your helpers ---
from dynamic_crl.helix_utils.helix_submitit import run_with_submitit
from dynamic_crl.src.dmc_envs.dc_envs.carl_dm_quadruped import CARLDmcQuadrupedEnv as dQuadEnv

from gymnasium.wrappers import FlattenObservation

# --- train contexts ---
# TODO: not sure how changing gravity affects observation normalization...
TRAIN_CONTEXTS = {
    0: {'gravity': 9.8},
    1: {'gravity': 15.0},
    2: {'gravity': 5.0},
    #0: {'gravity': 9.8},
    #1: {'gravity': 9.8},
}

EVAL_CONTEXTS = {
    0: {'gravity': 9.8},
    #0: {"gravity": 9.8},
    #1: {"gravity": 9.8},
}

OBS_CONTEXT_FEATURES = ['gravity'] #["gravity"]

# TODO:
# create a dynamic scheduler for gravity

DC_UPDATERS = {} # no dynamics changes for now

# custom flattening wrapper
from gymnasium import spaces
from gymnasium.spaces import flatdim

class MyFlattenWrapper(FlattenObservation):
    def __init__(self, env):

        # if 'context' is empty Dict(), remove it from the obs space
        if flatdim(env.observation_space["context"]) == 0:
            print("Removing empty 'context' from obs space")
            # we need to remove it from Box space
            old_space = env.observation_space
            new_spaces = {k: v for k, v in old_space.spaces.items() if k != 'context'}
            env.observation_space = spaces.Dict(new_spaces)
            print(f"New obs space: {env.observation_space}")

        super().__init__(env)
        # print(f"Wrapped obs space: {self.observation_space}")

# =========================
# Eval: pull StayStill metrics to TB
# =========================
class EvalInfoLogger(BaseCallback):
    """
    Every `every_steps`, build a fresh eval env via `make_eval_env()`,
    run `n_episodes` deterministic rollouts, collect info['eval_log'] at LAST,
    and log eval/* scalars to TensorBoard. (Tailored for StayStill.)
    """
    def __init__(self, make_eval_env: Callable[[], Any], every_steps: int = 20000, n_episodes: int = 10):
        super().__init__()
        self.make_eval_env = make_eval_env
        self.every_steps = int(every_steps)
        self.n_episodes = int(n_episodes)

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.every_steps != 0:
            return True

        eval_env = self.make_eval_env()

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

        def _mean(key):
            vals = [ep.get(key) for ep in per_ep if ep.get(key) is not None]
            return float(np.mean(vals)) if vals else np.nan

        # quadruped metrics -> 'move', 'upright', 'vx', 'upright_raw'
        move_reward = _mean("move")
        upright_reward = _mean("upright")
        vx = _mean("vx")
        upright_raw = _mean("upright_raw")

        self.logger.record("eval/move_reward", move_reward)
        self.logger.record("eval/upright_reward", upright_reward)
        self.logger.record("eval/vx", vx)
        self.logger.record("eval/upright_raw", upright_raw)

        self.logger.dump(self.model.num_timesteps)

        try:
            eval_env.close()
        except Exception:
            pass
        return True


# =========================
# Main training (runs inside allocation)
# =========================
def ppo_train_quadruped(results_root: Path) -> Tuple[float, float]:
    """
    Train PPO to hold the 'stand' keyframe using our DMC->Gym wrapper.
    Returns (mean_eval_return, std_eval_return).
    """
    version = "gravity_triplet_ctx"

    # --------- fixed settings you can edit here ---------
    SEED = 1
    TIMESTEPS = 5_000_000

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(SEED)
    except Exception:
        pass

    DEVICE = "cpu"  # PPO+MLP is usually faster on CPU

    # Eval/checkpoints
    EVAL_EPISODES = 10
    EVAL_FREQ = 10_000
    SAVE_FREQ = 250_000

    # env params
    time_limit_train = 6.0  # seconds
    EASY_INIT = True  # start in easy poses (not random)

    # Directory layout inside results_root
    logdir = results_root / f"ppo_quad_run_{version}"
    (logdir / "best").mkdir(parents=True, exist_ok=True)
    (logdir / "ckpts").mkdir(parents=True, exist_ok=True)

    # Auto-enable TensorBoard if present
    try:
        import tensorboard as _tb  # noqa: F401
        tb_dir = str(logdir)
    except Exception:
        tb_dir = None

    # ---------- Build envs ----------
    train_thunk = lambda: Monitor(MyFlattenWrapper( # theser two wrappers are for PPO compatibility
        dQuadEnv(
            render_mode="human",
            contexts=TRAIN_CONTEXTS,
            dc_updaters=DC_UPDATERS,
            quad_env_kwargs={"scale_toe_fric": False},  # whether to scale toe friction along with floor friction
            environment_kwargs={"flat_observation": True},
            task_kwargs={"time_limit": time_limit_train, "random": 10, "easy_init": EASY_INIT},
            obs_context_features=OBS_CONTEXT_FEATURES,
        )
    ))

    vec_env = DummyVecEnv([train_thunk])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    eval_thunk = lambda: Monitor(MyFlattenWrapper(
        dQuadEnv(
            render_mode=None,
            contexts=EVAL_CONTEXTS,
            dc_updaters=DC_UPDATERS,
            quad_env_kwargs={"scale_toe_fric": False},  # whether to scale toe friction along with floor friction
            environment_kwargs={"flat_observation": True},
            task_kwargs={"time_limit": 10.0, "random": 1000,
                         "is_evaluation": True, "easy_init": EASY_INIT},
            obs_context_features=OBS_CONTEXT_FEATURES,
        )
    ))

    eval_env = DummyVecEnv([eval_thunk])
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=True,
    )
    eval_env.obs_rms = vec_env.obs_rms  # sync stats

    # ---------- Model ----------
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.045,
        ent_coef=5e-3,
        vf_coef=0.7,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ELU,
            log_std_init=0.6,
            ortho_init=False,
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
    eval_cb_std = EvalInfoLogger(
        make_eval_env=eval_thunk,
        every_steps=EVAL_FREQ,
        n_episodes=EVAL_EPISODES
    )

    callbacks = [eval_cb, ckpt_cb, eval_cb_std] # not usingf eval_cb_std for now

    # ---------- Train ----------
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

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
    results_root_path = Path("/home/martin/Code/Thesis/dynamic_crl/src/dmc_envs/dc_envs/test_carl/run_logs/quadraped")
    results_root_path.mkdir(parents=True, exist_ok=True)
    return ppo_train_quadruped(results_root_path)


if __name__ == "__main__":
    version_suffix = "gravity_triplet_ctx"
    run_with_submitit(
        main_fn=ppo_main_wrapped,
        submit=None,
        ws_name="c_rl-exp",
        cpu=True,
        minutes=1200,
        cpus=2,
        mem_gb=8,
        gpu_type=None,
        part_cpu="cpu-single",
        part_gpu="gpu-single",
        link_target_subdir="runs",
        run_name=f"ppo_quadruped_run_{version_suffix}",
    )
