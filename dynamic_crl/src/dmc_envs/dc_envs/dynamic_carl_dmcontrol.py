from __future__ import annotations

from gymnasium import spaces

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.utils.types import Contexts

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../../..'))

from dynamic_crl.src.dmc_envs.loader import load_dmc_env
from dynamic_crl.src.dmc_envs.mj_gym_wrapper import MujocoToGymWrapper

# the environment that you load needs to be capable of handling dynamic contexts
class CARLDmcEnv_our(CARLEnv):
    """
    General class for the dm-control environments.

    Meta-class to change the context for the environments.

    Parameters
    ----------
    domain : str
        Dm-control domain that should be loaded.
    task : str
        Task within the specified domain.

    For descriptions of the other parameters see the parent class CARLEnv.

    Raises
    ------
    NotImplementedError
        Dict observation spaces are not implemented for dm-control yet.
    """

    def __init__(
        self,
        contexts: Contexts | None = None,
        obs_context_features: list[str] | None = None,
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        environment_kwargs: dict = {"flat_observation": True},
        task_kwargs: dict = None,
        **kwargs,
    ):
        # store these since resets for static context do lose the settings otherwise
        self.task_kwargs = task_kwargs if task_kwargs is not None else {}
        self.env_kwargs = environment_kwargs

        # Store evaluation mode
        self._is_evaluation = self.task_kwargs.get("is_evaluation", False)

        # TODO can we have more than 1 env?
        env = load_dmc_env(
            domain_name=self.domain,
            task_name=self.task,
            context={},
            task_kwargs=task_kwargs, # any task-specific args
            environment_kwargs=environment_kwargs, # e.g. time limit
        )
        env = MujocoToGymWrapper(env, eval_mode=self._is_evaluation)

        # this is the observation space without context features
        env.observation_space = spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            dtype=env.observation_space.dtype,
        )

        # now the environment is reinitialized for CARLEnv
        # -> self.observation_space is updated
        # -> self._add_context_to_state(state) will be called in reset() and step() -> dict('obs', 'context')
        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )

        # TODO: if you want to flatten the observation space, now is the time to do it
        # self.observation_space = spaces.flatten_space(self.observation_space)

        # TODO check gaussian noise on context features
        self.whitelist_gaussian_noise = list(
            self.get_context_features().keys()  # type: ignore
        )  # allow to augment all values

    def _update_context(self) -> None:
        # for initialization of the env (first update to be called upon reset by CARLEnv)
        env = load_dmc_env(
            domain_name=self.domain,
            task_name=self.task,
            context=self.context,
            task_kwargs=self.task_kwargs,
            environment_kwargs=self.env_kwargs,
        )
        self.env = MujocoToGymWrapper(env, eval_mode=self._is_evaluation)

    def render(self):
        return self.env.render(mode="rgb_array")
