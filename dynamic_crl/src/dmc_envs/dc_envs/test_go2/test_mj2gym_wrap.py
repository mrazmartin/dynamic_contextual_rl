# tests/test_mj_gym_wrapper.py
from __future__ import annotations
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../../..')
sys.path.append(project_root)

from typing import Optional, Tuple
import collections
import numpy as np
import gymnasium as gym

import dm_env
from dm_env import StepType
from dm_control.suite import base  # only for typing, your make_env returns dm_env.Environment

# import your stuff
# from your_module import make_env, StandUp, MujocoToGymWrapper
from dynamic_crl.src.dmc_envs.dc_envs.dmc_tasks.dmc_go2 import StandUp, make_env
from dynamic_crl.src.dmc_envs.mj_gym_wrapper import MujocoToGymWrapper

FLAT_KEY = "observations"

def _flatten_like_dm(obs: dict) -> np.ndarray:
    """Match dm_control.flatten_observation ordering: OrderedDict order else sorted keys."""
    if isinstance(obs, collections.OrderedDict):
        keys = list(obs.keys())
    else:
        keys = sorted(obs.keys())
    parts = [np.asarray(obs[k]).ravel() for k in keys]
    return np.concatenate(parts, axis=0)


def _pose_summary(physics, tag: str, verbose: bool = True):
    """
    Return (base_height, up_alignment, roll, pitch) tuple and print if verbose.
    """
    bh = physics.base_height()
    up = physics.up_alignment()
    roll, pitch = physics.roll_pitch()
    if verbose:
        print(f"[{tag}] base_height={bh:.3f}, up={up:.3f}, roll={roll:.3f}, pitch={pitch:.3f}")
    return bh, up, roll, pitch


# ---------------------------
# 1) Parity test: resets with dm_env vs wrapped Gym env
# ---------------------------

def test_wrapper_resets(
    *,
    task: Optional[base.Task] = None,
    time_per_episode: float = 1.0,
    num_episodes: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> bool:
    """
    Mirrors your dm_control test but runs via the Gym wrapper.
    Verifies that after each reset, physics returns to the initial pose.
    """
    # Build raw dm_control env then wrap it
    dm = make_env(task=task, time_limit=time_per_episode)
    env = MujocoToGymWrapper(dm)

    rng = np.random.default_rng(seed)
    a_space: gym.spaces.Box = env.action_space

    # --- Initial reset ---
    obs, info = env.reset(seed=seed)
    physics = env.env.physics  # access underlying physics via wrapper
    if verbose:
        print("\n=== After initial reset (wrapped) ===")
    initial_pose = _pose_summary(physics, "reset[0]", verbose)

    for ep in range(num_episodes):
        if verbose:
            print(f"\n--- Episode {ep} rollout ({time_per_episode}s, random policy) ---")
        while True:
            a = rng.uniform(a_space.low, a_space.high, size=a_space.shape).astype(a_space.dtype)
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                if verbose:
                    step_count = dm._step_count if hasattr(dm, "_step_count") else "?"
                    print(f"Episode ended at {step_count} * {dm.control_timestep()} s")
                    final_pose = _pose_summary(physics, f"end_ep{ep}", verbose)
                    print(f"Final pose vs initial [b_height, up_alignment, roll, pitch]:\n{final_pose}\nvs.\n{initial_pose}\n")
                break

        # Reset and compare pose
        obs, info = env.reset()
        after_reset_pose = _pose_summary(physics, f"reset[{ep+1}]", verbose)
        if after_reset_pose != initial_pose:
            if verbose:
                print(f"✗ MISMATCH after reset[{ep+1}] vs initial reset[0]")
            return False

    if verbose:
        print("\nOK ✓ wrapper resets return to the task's initial pose each episode.")
    return True


# ---------------------------
# 2) API contract & semantics tests
# ---------------------------

def test_wrapper_api_contract(task: Optional[base.Task] = None, time_per_episode: float = 1.0, seed: int = 0) -> None:
    """Basic contract: shapes, dtypes, return tuple order, and discount/termination mapping."""
    dm = make_env(task=task, time_limit=time_per_episode)
    env = MujocoToGymWrapper(dm)

    # Reset
    obs, info = env.reset(seed=seed)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert env.observation_space.shape == obs.shape
    assert obs.dtype == env.observation_space.dtype
    assert "discount" in info and "step_type" in info

    # Step with slightly out-of-bounds action -> wrapper clips
    a = np.ones(env.action_space.shape, dtype=env.action_space.dtype) * 1e9
    obs2, reward, terminated, truncated, info2 = env.step(a)
    assert isinstance(reward, float)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "discount" in info2 and "step_type" in info2

    # dm_control termination semantics:
    # terminated <=> (discount is not None) <=> StepType.LAST
    assert (terminated == (info2["discount"] is not None)) == (info2["step_type"] == StepType.LAST)


def test_wrapper_observation_equivalence(
    task: Optional[base.Task] = None,
    time_per_episode: float = 1.0,
    seed: int = 0,
) -> None:
    """
    If dm_control env already provides FLAT_KEY='observations', we use it.
    Otherwise we flatten deterministically and the result must match spec shape.
    """
    dm = make_env(task=task, time_limit=time_per_episode)
    env = MujocoToGymWrapper(dm)

    # Reset once and compare what wrapper returns to a manual flatten
    ts = dm.reset()
    if isinstance(ts.observation, dict) and FLAT_KEY in ts.observation:
        ref = np.asarray(ts.observation[FLAT_KEY]).ravel()
    else:
        ref = _flatten_like_dm(ts.observation)

    obs_wrapped, _ = env.reset(seed=seed)

    # Shapes must match wrapper space, and wrapper obs equals reference up to dtype cast
    assert obs_wrapped.shape == env.observation_space.shape
    np.testing.assert_allclose(
        obs_wrapped.astype(np.float64, copy=False),
        ref.astype(np.float64, copy=False),
        atol=0.0, rtol=0.0,
    )

# ---------------------------
# Example manual run (not pytest)
# ---------------------------

if __name__ == "__main__":
    # Example usage mirroring your call:
    starter_pose = "stand"

    ok = test_wrapper_resets(
        task=StandUp(start_key=starter_pose, min_alive_height=0.05, max_tilt_deg=60.0),
        time_per_episode=100.0,
        num_episodes=2,
        seed=42,
        verbose=True,
    )
    print("Resets OK?", ok)

    # Quick smoke tests
    test_wrapper_api_contract(task=StandUp(start_key=starter_pose), time_per_episode=5.0, seed=0)
    test_wrapper_observation_equivalence(task=StandUp(start_key=starter_pose), time_per_episode=5.0, seed=0)
    print("Wrapper tests passed ✓")
