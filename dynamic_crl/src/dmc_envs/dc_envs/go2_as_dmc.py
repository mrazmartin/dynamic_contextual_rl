# go2_as_dmc.py
from __future__ import annotations
import os, sys
from typing import Callable, Optional

import numpy as np
from dm_control import viewer

# Ensure project root on path (adjust levels if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../..')
sys.path.append(project_root)

from carl.context.context_space import UniformFloatContextFeature, CategoricalContextFeature

# make sure dm_control can find your domain module
from dynamic_crl.src.dmc_envs.dc_envs.dmc_tasks import dmc_go2  # noqa: E402
from dynamic_crl.src.dmc_envs.dc_envs.dynamic_carl_dmcontrol import CARLDmcEnv_our

speed_presets = {
    "stop": 0.0,
    "crawl": 0.3,
    "walk": 1.0,
    "trot": 2.0,
    "run": 3.0,
}

# already contextualized, domain and task are pre-set
class CARLDmcGo2Env(CARLDmcEnv_our):
    domain = "go2"
    task = "stand_up"
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, *args, render_mode=None, **kwargs):
        self._render_mode = render_mode
        self._viewer_policy: Optional[Callable] = None
        self._dynamic_updaters = kwargs["dc_updaters"] if ("dc_updaters" in kwargs) else {}
        task_kwargs = kwargs.pop("task_kwargs", {}) if ("task_kwargs" in kwargs) else {}
        environment_kwargs = kwargs.pop("environment_kwargs", {}) if ("environment_kwargs" in kwargs) else {}
        super().__init__(*args, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs, **kwargs)
        self._step_count = 0

    @staticmethod
    def get_context_features():
        return {
            # physics and environment
            "gravity_z": UniformFloatContextFeature(
                name="gravity_z", lower=-1.0, upper=-0.1, default_value=-0.1
            ),
            "ground_friction": UniformFloatContextFeature(
                name="ground_friction", lower=0.2, upper=1.2, default_value=0.8
            ),
            # payload on base
            "payload_mass": UniformFloatContextFeature(
                name="payload_mass", lower=0.0, upper=10.0, default_value=0.0
            ),
            # task and runtime conditions
            "desired_speed": UniformFloatContextFeature(
                "desired_speed", lower=0.0, upper=3.0, default_value=1.0
            ),
            "speed_preset": CategoricalContextFeature(
                "speed_preset", choices=list(speed_presets.keys()), default_value="walk"
            ),
            "latency_steps": CategoricalContextFeature(
                name="latency_steps", choices=[0, 1, 2, 3, 4, 5], default_value=0
            ),
        }

    # latency and dynamic hook setup after CARLEnv has selected context and reloaded dm_control
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # pick up latency from current context
        ctx = getattr(self, "context", {}) or {}
        self._latency_steps = int(ctx.get("latency_steps", 0))
        # self._action_buffer.clear()
        self._step_count = 0

        return obs, info

    def _update_dynamic_context(self, context_to_update) -> None:
        if 'gravity_z' in context_to_update:
            self.env.env.physics.dynamic_update_gravity(new_gz = context_to_update['gravity_z'])
        if 'payload_mass' in context_to_update:
            self.env.env.physics.dynamic_update_payload_mass(new_mass = context_to_update['payload_mass'])

    def step(self, action):
        # check if we have a dynamic updater, if yes, run their update
        ctx_to_update = {}
        for key, dyn_updater in self._dynamic_updaters.items():
            self.context[key] = dyn_updater(self._step_count)
            ctx_to_update[key] = self.context[key]
        self._update_dynamic_context(ctx_to_update)
        # simulate control latency
        if self._latency_steps > 0:
            self._action_buffer.append(np.asarray(action))
            if len(self._action_buffer) <= self._latency_steps:
                eff_action = np.zeros_like(action)
            else:
                eff_action = self._action_buffer.pop(0)
        else:
            eff_action = action

        self._step_count += 1

        return super().step(eff_action)

    # ==== for rendering only ====
    # viewer helpers
    def set_viewer_policy(self, policy_fn):
        self._viewer_policy = policy_fn

    def _random_policy(self):
        spec = self.env.env.action_spec()
        lo = np.asarray(spec.minimum, dtype=np.float32)
        hi = np.asarray(spec.maximum, dtype=np.float32)
        shape = spec.shape
        def policy_fn(ts):
            return np.random.uniform(lo, hi, size=shape).astype(np.float32)
        return policy_fn

    def render(self):
        if self._render_mode == "human":
            policy = self._viewer_policy or self._random_policy()
            hooked = _ViewerHookEnv(self)
            try:
                viewer.launch(hooked, policy=policy, title="Go2 Viewer")
            except TypeError:
                viewer.launch(hooked)
            return None

        # rgb array path
        try:
            physics = self.env.env.physics
            # try a named camera first
            try:
                cam_id = physics.model.name2id("camera", "chase")
            except Exception:
                cam_id = 0
            frame = physics.render(height=480, width=640, camera_id=cam_id)
            return frame
        except Exception:
            # fallback through the gym wrapper
            return self.env.render(mode="rgb_array")

# ---- viewer hook env ---
# add near your proxy definition
import dm_env

class _ViewerHookEnv(dm_env.Environment):
    """Proxy so dm_control.viewer calls our dynamic update before stepping."""
    def __init__(self, outer):
        self._outer = outer
        self._env = outer.env.env  # underlying dm_control Environment

    # --- required by the viewer ---
    @property
    def physics(self):
        # delegate to the wrapped dm_control env
        return self._env.physics

    # (optional but handy) delegate unknown attrs to the underlying env
    def __getattr__(self, name):
        return getattr(self._env, name)

    # ---- dm_env.Environment interface ----
    def reset(self):
        ts = self._env.reset()
        try:
            self._outer._step_count = 0
        except Exception:
            pass
        return ts

    def step(self, action):
        # run your per-step dynamic context hook BEFORE stepping
        try:
            ctx = {}
            for key, dyn_updater in self._outer._dynamic_updaters.items():
                ctx[key] = dyn_updater(self._outer._step_count)
            if ctx:
                self._outer._update_dynamic_context(ctx)
        except Exception:
            pass

        ts = self._env.step(action)
        try:
            self._outer._step_count += 1
        except Exception:
            pass
        return ts

    # pass-throughs the viewer/policy might use
    def action_spec(self):      return self._env.action_spec()
    def observation_spec(self): return self._env.observation_spec()
    def time_step(self):        return self._env.time_step()


if __name__ == "__main__":

    # example use case of CARL + dmc + DynamicContext

    contexts = {
        0: {
            "payload_mass": 0.0,
        },
        1: {
            "payload_mass": 10000.0,
        },
    }

    env = CARLDmcGo2Env(
        render_mode="human",
        contexts=contexts,
        dc_updaters={
            "gravity_z": lambda step: -9.8 if step >= 500 else -0.5,   # switch gravity at t=600
            #"payload_mass": lambda step: 10000 if step >= 200 else 2.0,
        }
    )
    env.reset()
    env.render()
    env.reset()
    env.render()
