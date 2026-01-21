import numpy as np

import os, sys
# Ensure project root on path (adjust levels if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../..')
sys.path.append(project_root)
from dm_control import viewer
from typing import Optional, Callable

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from dynamic_crl.src.dmc_envs.dc_envs.dynamic_carl_dmcontrol import CARLDmcEnv_our

# dynamic context verion
class CARLDmcQuadrupedEnv(CARLDmcEnv_our):
    domain = "quadruped"
    task = "walk_context"
    metadata = {"render_modes": []}

    def __init__(self, *args, render_mode=None, **kwargs):
        self._render_mode = render_mode
        self._viewer_policy: Optional[Callable] = None
        self._dynamic_updaters = kwargs["dc_updaters"] if ("dc_updaters" in kwargs) else {}

        # obs_context_features in kwargs can limit which context features are included in obs
        super().__init__(*args, **kwargs)
        self._step_count = 0

        # unload environment kvargs
        quad_env_kwargs = kwargs.get("quad_env_kwargs", {})

        # setup for dynamic edits
        self.floor_geom_names = [name for name in self.env.env.physics.named.model.geom_friction.axes[0].names if "floor" in name]
        self.toes_geom_names = [name for name in self.env.env.physics.named.model.geom_friction.axes[0].names if "toe" in name]
        self.scale_toes_friction = quad_env_kwargs.get("scale_toe_fric", False)  # whether to scale toes friction along with floor friction
        self.wind_body_name = "torso"  # body to apply wind forces to
        self._is_pushed = bool(self._dynamic_updaters.get('d_wind', None) is not None)
        self._dynamic_defaults = None

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=0.1, upper=np.inf, default_value=9.81
            ),
            "friction_torsional": UniformFloatContextFeature(
                "friction_torsional", lower=0, upper=np.inf, default_value=1.0
            ),
            "friction_rolling": UniformFloatContextFeature(
                "friction_rolling", lower=0, upper=np.inf, default_value=1.0
            ),
            "friction_tangential": UniformFloatContextFeature(
                "friction_tangential", lower=0, upper=np.inf, default_value=1.0
            ),
            "timestep": UniformFloatContextFeature(
                "timestep", lower=0.001, upper=0.1, default_value=0.005
            ),
            "joint_damping": UniformFloatContextFeature(
                "joint_damping", lower=0.0, upper=np.inf, default_value=1.0
            ),
            "joint_stiffness": UniformFloatContextFeature(
                "joint_stiffness", lower=0.0, upper=np.inf, default_value=0.0
            ),
            "actuator_strength": UniformFloatContextFeature(
                "actuator_strength", lower=0.0, upper=np.inf, default_value=1.0
            ),
            "density": UniformFloatContextFeature(
                "density", lower=0.0, upper=np.inf, default_value=0.0
            ),
            "viscosity": UniformFloatContextFeature(
                "viscosity", lower=0.0, upper=np.inf, default_value=0.0
            ),
            "geom_density": UniformFloatContextFeature(
                "geom_density", lower=0.0, upper=np.inf, default_value=1.0
            ),
            "wind_x": UniformFloatContextFeature(
                "wind_x", lower=-np.inf, upper=np.inf, default_value=0.0
            ),
            "wind_y": UniformFloatContextFeature(
                "wind_y", lower=-np.inf, upper=np.inf, default_value=0.0
            ),
            "wind_z": UniformFloatContextFeature(
                "wind_z", lower=-np.inf, upper=np.inf, default_value=0.0
            ),
        }

    # latency and dynamic hook setup after CARLEnv has selected context and reloaded dm_control
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # pick up some context setting for new episode init
        # ctx = getattr(self, "context", {}) or {}
        # self._xxx = int(ctx.get("xxx", 0))

        # self._action_buffer.clear()
        self._step_count = 0

        return obs, info

    def _capture_dynamic_defaults(self):
        physics = self.env.env.physics
        self._dynamic_defaults = {
            "gravity": physics.model.opt.gravity[2],
            "floor_friction": {name: physics.named.model.geom_friction[name].copy()
                                for name in self.floor_geom_names},
            "actuator_gear0": physics.model.actuator_gear[:, 0].copy(),
            "dof_damping0": physics.model.dof_damping.copy(),  # length nv
        }

    def _update_dynamic_context(self, context_to_update) -> None:
        did_edit_model = False
        if self._dynamic_defaults is None:
            self._capture_dynamic_defaults()
        if 'gravity' in context_to_update:
            self.env.env.physics.model.opt.gravity[2] = context_to_update['gravity']
        if 'friction' in context_to_update:
            mu_slide = float(context_to_update["friction"])
            for name in self.floor_geom_names:
                if name in self.env.env.physics.named.model.geom_friction.axes[0].names:
                    # Only sliding (index 0). Leave torsional/rolling as-is.
                    fr = self.env.env.physics.named.model.geom_friction[name]
                    fr[0] = mu_slide
            # for now we only edit floor friction
            if self.scale_toes_friction:
                for name in self.toes_geom_names:
                    if name in self.env.env.physics.named.model.geom_friction.axes[0].names:
                        # Only sliding (index 0). Leave torsional/rolling as-is.
                        fr = self.env.env.physics.named.model.geom_friction[name]
                        fr[0] = mu_slide
            did_edit_model = True
        if 'actuator_strength' in context_to_update:
            scale = float(context_to_update["actuator_strength"])
            self.env.env.physics.model.actuator_gear[:, 0] = self._dynamic_defaults["actuator_gear0"] * scale
            did_edit_model = True
        if 'joint_damping' in context_to_update:
            k = float(context_to_update["joint_damping"])
            # some of these start at 0.0, so cant expect to scale them all
            self.env.env.physics.model.dof_damping[:] = self._dynamic_defaults["dof_damping0"] * k  # length nv
            did_edit_model = True
        if 'd_wind' in context_to_update: # this is dynamic wind, not to be confused with static wind in context
            # this only work if the xml model has wind enabled
            self.context['wind_x'] = float(context_to_update.get('d_wind')[0])
            self.context['wind_y'] = float(context_to_update.get('d_wind')[1])
            self.context['wind_z'] = float(context_to_update.get('d_wind')[2])
            # wind is applied in step()
        
        if did_edit_model:
            self.env.env.physics.forward()  # ensure changes take effect

    def apply_push(self, physics, body_name='torso', force_xyz=(2000.0, 0.0, 0.0)):
        # World-frame force (N) and torque (N*m); resets to zero each step.
        physics.named.data.xfrc_applied[body_name] = (*force_xyz, 0.0, 0.0, 0.0)

    def step(self, action):
        # check if we have a dynamic updater, if yes, run their update
        ctx_to_update = {}
        for key, dyn_updater in self._dynamic_updaters.items():
            self.context[key] = dyn_updater(self._step_count)
            ctx_to_update[key] = self.context[key]
        self._update_dynamic_context(ctx_to_update)
        if self._is_pushed:
            self.apply_push(self.env.env.physics, body_name="torso",
                            force_xyz=(self.context.get('wind_x', 0.0),
                                       self.context.get('wind_y', 0.0),
                                       self.context.get('wind_z', 0.0)))

        self._step_count += 1

        return super().step(action)
    
    # for rendering
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
    
    def _zero_policy(self):
        spec = self.env.env.action_spec()
        shape = spec.shape
        def policy_fn(ts):
            return np.zeros(shape, dtype=np.float32)
        return policy_fn

    def scale_render_cam(self):
        self._cam_rescaled = True
        print("[info] Scaling render camera closer")
        set_camera_distance(self.env.env.physics, "global", scale=0.2, fovy_scale=0.5)

    def render(self):
        if self._render_mode == "human":
            policy = self._viewer_policy or self._zero_policy()
            hooked = _ViewerHookEnv(self)
            try:
                viewer.launch(hooked, policy=policy, title="Dynamic Viewer")
            except TypeError:
                viewer.launch(hooked)
            return None

        # rgb array path
        try:
            physics = self.env.env.physics
            # once scale the global camera (fails when it is not present)
            try:
                if not hasattr(self, "_cam_rescaled"):
                    self.scale_render_cam()
            except Exception:
                raise ValueError("Likely 'global' camera is not present in the model.")
            # try a named camera first
            try:
                cam_id = physics.model.name2id("camera", "global")
            except Exception:
                cam_id = 0
            frame = physics.render(height=480, width=640, camera_id=cam_id)
            return frame
        except Exception:
            # fallback through the gym wrapper
            return self.env.render(mode="rgb_array")

# =========================
# get closer camera
def set_camera_distance(physics, cam_name: str, scale: float = 1.5, fovy_scale: float | None = None):
    """
    Moves a fixed camera farther by scaling its local position vector.
    Optionally adjusts FOV (bigger fovy => looks farther).
    """
    # move camera away along its current direction (scales pos in parent frame)
    pos = physics.named.model.cam_pos[cam_name].copy()
    physics.named.model.cam_pos[cam_name] = pos * float(scale)

    if fovy_scale is not None:
        fovy = float(physics.named.model.cam_fovy[cam_name])
        physics.named.model.cam_fovy[cam_name] = fovy * float(fovy_scale)

# ---- viewer hook env ---
# add near your proxy definition
import dm_env

class _ViewerHookEnv(dm_env.Environment):
    """Proxy so dm_control.viewer calls our dynamic update before stepping."""
    def __init__(self, outer):
        # of outer.observation_space is gym.spaces.Dict, we need to flatten it for viewer
        from gymnasium.spaces import flatten_space
        self._outer = outer
        self._outer.observation_space = flatten_space(self._outer.observation_space)

        # this will get the underlying dm_control Environment -> it doesnt not have the concated observation and context
        self._env = self._outer.env.env  # underlying dm_control Environment

        # first modify how the obs spec is computed <- not needed
        # self._env.observation_spec = self.observation_spec  # flattened spec

        # modify how get_observation is computed -> requires modification to the task
        # MIND THE ORDER OF CONTEXT AND OBSERVATIONS -> CTX GOES FIRST, ORDER IN CTX MATTERS!!!
        self._env._task.get_observation_vanilla = self._env._task.get_observation
        self._env._task.get_observation = self.modified_task_get_observation

        self.reward = 0.0

    def modified_task_get_observation(self, physics):
        # get the default observations
        observations = self._env._task.get_observation_vanilla(physics)
        # get the context features
        from collections import OrderedDict
        ctx_dict = OrderedDict()
        if hasattr(self._outer, "context"):
            for key in self._outer.context:
                ctx_dict[key] = np.array(self._outer.context[key])
        
        # CONTEXT GOES FIRST WHEN APPENDING IT TO OBSERVATIONS
        merged = ctx_dict | observations
        return merged

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
        if self.reward:
            print(f"[info] Episode ended with reward {self.reward:.3f}")
        self.reward = 0.0
        try:
            self._outer._step_count = 0
        except Exception:
            pass
        return ts
    
    def apply_push(self, physics, body_name='torso', force_xyz=(2000.0, 0.0, 0.0)):
        # World-frame force (N) and torque (N*m); resets to zero each step.
        physics.named.data.xfrc_applied[body_name] = (*force_xyz, 0.0, 0.0, 0.0)

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

        self.apply_push(self._env.physics, body_name="torso",
                        force_xyz=(self._outer.context.get('wind_x', 0.0),
                                   self._outer.context.get('wind_y', 0.0),
                                   self._outer.context.get('wind_z', 0.0)))

        ts = self._env.step(action)
        self.reward += float(ts.reward or 0.0)
        try:
            self._outer._step_count += 1
        except Exception:
            pass
        return ts

    # pass-throughs the viewer/policy might use
    def action_spec(self):      return self._env.action_spec()
    def observation_spec(self):
        # redo the flattening here
        from dm_control.rl.control import _spec_from_observation
        from collections import OrderedDict

        # get the default observations
        observations = self._env.task.get_observation(self._env.physics)
        # get the context features
        ctx_dict = OrderedDict()
        if hasattr(self._outer, "context"):
            for key in self._outer.context:
                ctx_dict[key] = np.array(self._outer.context[key])

        # CONTEXT GOES FIRST WHEN APPENDING IT TO OBSERVATIONS
        merged = ctx_dict | observations
        spec = _spec_from_observation(merged)

        return spec
    
    def time_step(self):        return self._env.time_step()



# =========================
# best example of dynamic + contextual + dmc Environment, which can do it all :)
# =========================
if __name__ == "__main__":

    # example use case of CARL + dmc + DynamicContext

    dynamic_fns_g = {
        #"gravity": lambda step: -9.8 if step >= 50 else 0.1 ,  # switch gravity every 500 steps
        "d_wind": lambda step: [200, 0, 0] if step >= 100 else [0, 0, 0], 
        #"joint_damping": lambda step: 0.0 if step >= 150 else 0.1, # switch damping at t=150
        "friction": lambda step: 0.00001 if step >= 300 else 2.0, # switch friction at t=150
    }

    # default context
    contexts = {
        0: {"gravity": 0.1,
            },
        1: {"gravity": 9.8,
            "wind_x": 0.0, "wind_y": 0.0, "wind_z": 0.0,
          }
    }

    env = CARLDmcQuadrupedEnv(
        render_mode="human",
        contexts=contexts,
        dc_updaters=dynamic_fns_g,
        quad_env_kwargs={"scale_toe_fric": True},  # whether to scale toe friction along with floor friction
        environment_kwargs={"flat_observation": True},
        task_kwargs={"time_limit": 10.0, "random": 10, "is_evaluation": True},
    )
    env.reset() # this will restart the env (and possibly the task kwargs)
    env.set_viewer_policy(env._zero_policy())  # or env._random_policy()
    env.render()
    env.reset()
    env.render()
