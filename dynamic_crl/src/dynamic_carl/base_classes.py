import numpy as np
# wrapper for limiting the contextual features --> dont include observations of static context
import gymnasium as gym
from gymnasium import spaces

class PartialContextWrapper(gym.ObservationWrapper):
    """
    Wrapper for partially observing the context in a Gym environment.
    -> only needed if the observation space of the env was poorly initialized
        and includes also the unobserved context features.
    """
    def __init__(self, env, selected_context_keys):
        super().__init__(env)
        self.selected_keys = list(selected_context_keys or [])

        assert isinstance(env.observation_space, spaces.Dict) and "obs" in env.observation_space.spaces, \
            "PartialContextWrapper expects a Dict observation space with key 'obs'"
        if self.selected_keys:
            assert "context" in env.observation_space.spaces, \
                "selected_context_keys provided but base env has no 'context' space"

        obs_space: spaces.Box = env.observation_space["obs"]
        obs_low  = np.asarray(obs_space.low, dtype=np.float32).reshape(-1)
        obs_high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)

        if self.selected_keys:
            ctx_space: spaces.Dict = env.observation_space["context"]
            ctx_lows  = [np.asarray(ctx_space.spaces[k].low,  dtype=np.float32).reshape(-1)[0] for k in self.selected_keys]
            ctx_highs = [np.asarray(ctx_space.spaces[k].high, dtype=np.float32).reshape(-1)[0] for k in self.selected_keys]
            low  = np.concatenate([obs_low,  np.asarray(ctx_lows,  dtype=np.float32)])
            high = np.concatenate([obs_high, np.asarray(ctx_highs, dtype=np.float32)])
        else:
            low, high = obs_low, obs_high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs_dict):
        obs = np.asarray(obs_dict["obs"], dtype=np.float32).reshape(-1)
        if not self.selected_keys:
            return obs
        if "context" not in obs_dict:
            raise RuntimeError("PartialContextWrapper received an observation without 'context'. "
                               "Do not use this wrapper when observe_context_mode is 'none'.")
        # if the context is empty, make it an empty vector
        if not obs_dict["context"]:
            context = np.empty((0,), dtype=np.float32)
        else:
            context = np.asarray([obs_dict["context"][k] for k in self.selected_keys], dtype=np.float32).reshape(-1)
        return np.concatenate([obs, context], dtype=np.float32)

    # State/Context tracker Episode ID setter/getter passthrough
    def get_sc_ep(self):
        return self.env.get_sc_ep()

    def set_sc_ep(self, value: int):
        return self.env.set_sc_ep(value)

