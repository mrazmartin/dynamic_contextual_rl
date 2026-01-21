from typing import Any, Optional, Tuple, TypeVar, Union
import collections

import dm_env
import gymnasium as gym
import numpy as np
from dm_env import StepType
from gymnasium import spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

FLAT_KEY = "observations"  # dm_control FLAT_OBSERVATION_KEY convention

class MujocoToGymWrapper(gym.Env):
    """
    Tolerant dm_control -> Gymnasium wrapper.

    - If the env is built with flat_observation=True, it expects a single key
      'observations' holding a pre-flattened vector.
    - Otherwise, it flattens the observation dict itself:
        * If the dict is an OrderedDict, preserves that order.
        * Else, sorts keys alphabetically (matching dm_control behavior).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env: dm_env.Environment, *, eval_mode:bool=False) -> None:
        self.env = env
        self._eval_mode = bool(eval_mode)

        # ---------- Action space ----------
        a_spec = self.env.action_spec()
        self._act_min = np.asarray(a_spec.minimum, dtype=a_spec.dtype)
        self._act_max = np.asarray(a_spec.maximum, dtype=a_spec.dtype)
        self.action_space = spaces.Box(low=self._act_min, high=self._act_max, dtype=a_spec.dtype)

        # ---------- Observation space (from spec) ----------
        obs_spec = self.env.observation_spec()
        if not isinstance(obs_spec, dict):
            raise NotImplementedError("Non-dict observation specs are not supported.")

        if FLAT_KEY in obs_spec:  # dm_control flat mode or CARL-style
            vec_spec = obs_spec[FLAT_KEY]
            flat_size = int(np.prod(tuple(vec_spec.shape or (1,))))
            dtype = vec_spec.dtype
        else:
            # Sum sizes of all fields
            flat_size = 0
            dtypes = set()
            for spec in obs_spec.values():
                flat_size += int(np.prod(tuple(spec.shape or (1,))))
                dtypes.add(spec.dtype)
            # If mixed dtypes, we’ll cast to float32 below
            dtype = dtypes.pop() if len(dtypes) == 1 else np.float32

        # Favor float32 for SB3 unless spec already float32/64
        if dtype not in (np.float32, np.float64):
            dtype = np.float32
        self._obs_dtype = dtype

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=self._obs_dtype
        )

    # --------------- Gymnasium API ---------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options) # CARL calls super(MujocoToGymWrapper,self).reset(...)
        ts = self.env.reset()
        obs = self._extract_obs(ts) # unlike CARL, we can always flatten
        # dm_control (and CARL) semantics at reset: FIRST step, reward=None, discount=None
        info = {
            "discount": ts.discount,   # None at reset
            "step_type": ts.step_type, # StepType.FIRST
        }
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Clip to action spec bounds
        action = np.asarray(action, dtype=self.action_space.dtype) # make sure correct dtype
        action = np.clip(action, self._act_min, self._act_max) # make sure correct bounds

        ts = self.env.step(action) # take the action

        obs = self._extract_obs(ts) # flatten observation to 1D vector
        reward = float(ts.reward or 0.0) # extract reward from our action

        # dm_control termination semantics:
        is_last = (ts.step_type == StepType.LAST)
        if is_last and self.env._task.get_termination(self.env._physics) is None:
            truncated = True  # time limit truncation
            terminated = False
        else:
            truncated = False # set True only if you wrap with a time-limit - still gets the terminated signal only
            terminated = bool(is_last)

        info = {
            "discount": ts.discount,      # dm_control’s termination/continuation signal
            "step_type": ts.step_type,    # MID/FIRST/LAST for debugging
        }

        # attach eval summary ONLY for eval envs and only at episode end
        if self._eval_mode and is_last:
            eval_log = None
            try:
                task = getattr(self.env, "_task", None) or getattr(self.env, "task", None)
                if task is not None and hasattr(task, "_eval_log"):
                    eval_log = dict(task._eval_log)  # shallow copy
                    # print(f"[info] eval_log: {eval_log}")
            except Exception:
                eval_log = None
            info["eval_log"] = eval_log
        
        # return obs, reward, truncated, terminated, info # WHAT CARL WANTS
        return obs, reward, terminated, truncated, info # latest gym order

    def render(self, mode: str = "human", camera_id: int = 0, **kwargs: Any) -> np.ndarray:
        if mode == "human":
            raise NotImplementedError("Human rendering not implemented.")
        if mode == "rgb_array":
            return self.env.physics.render(camera_id=camera_id, **kwargs)
        raise NotImplementedError(f"Unsupported render mode: {mode!r}")

    # --------------- Internals ---------------

    def _extract_obs(self, ts: dm_env.TimeStep) -> np.ndarray:
        """Return a 1D contiguous vector matching observation_space and dtype."""
        ob = ts.observation
        if isinstance(ob, dict) and FLAT_KEY in ob:
            vec = np.asarray(ob[FLAT_KEY])
        else:
            # Generic flatten (mirror dm_control.flatten_observation)
            if not isinstance(ob, collections.abc.Mapping):
                raise ValueError("Expected mapping for timestep.observation.")
            if isinstance(ob, collections.OrderedDict):
                keys = ob.keys()
            else:
                keys = sorted(ob.keys())
            parts = [np.asarray(ob[k]).ravel() for k in keys]
            vec = np.concatenate(parts, axis=0)

        vec = np.ravel(vec)
        if vec.dtype != self._obs_dtype:
            vec = vec.astype(self._obs_dtype, copy=False)
        return np.ascontiguousarray(vec)
