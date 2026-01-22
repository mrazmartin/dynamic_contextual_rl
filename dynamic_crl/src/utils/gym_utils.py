from carl.envs import CARLCartPole as cartpole
from carl.envs import CARLBipedalWalker as walker

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

from dynamic_crl.src.utils.log_msgs import warn_msg, info_msg
import gymnasium as gym
import math

def get_gym_base_env(e):
    """
    Unwrapes nested Gym wrappers (CARL included).
    Use to directly modify the environment physics engines.
    """
    cur = e
    for _ in range(16):
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return getattr(cur, "unwrapped", cur)

def get_CARL_env(env):
    """
    Unwraps nested Gym wrappers (FlattenObservation, DummyVecEnv, etc.)
    but stops at CARL envs which define `.context`.
    """
    while hasattr(env, "env") and not hasattr(env, "context"):
        env = env.env
    return env

def get_ctx_env_from_dummy_vec_env(env):
    """
    Unwraps DummyVecEnv to get the base environment.
    """
    if hasattr(env, "envs"):
        return env.envs[0].env # we have DummyVecEnv - OurContextWrapper -> the env used for training
    else:
        raise ValueError("Provided environment is not a DummyVecEnv")

# cartpole
def _make_cp_ctx_accessor(attr: str):
    def getter(e):
        base = get_gym_base_env(e)
        return getattr(base, attr, None)
    def setter(e, v):
        base = get_gym_base_env(e)
        setattr(base, attr, float(v))
    return getter, setter

def cartpole_env_factory(contexts=None, render_mode=None, ctx_to_observe=None):
    """
    Factory function to create a CARL CartPole environment with optional rendering and seed.
    """

    if contexts is None:
        env = cartpole(render_mode=render_mode)
    else:
        env = cartpole(contexts=contexts, render_mode=render_mode, obs_context_features=ctx_to_observe)
    
    env.render_mode = render_mode

    # tiny hack for our dynamic context
    try:
        raw = env.env.unwrapped
        env.env._init_density = raw.masspole / raw.length
    except Exception:
        pass

    return env


# walker
from carl.envs import CARLBipedalWalker

def _get_attr(env, name: str):
    """
    Robust attribute fetch that:
      1) Reads from the base env (env.unwrapped) if present.
      2) Otherwise asks wrappers via get_wrapper_attr(name) without touching wrapper.<name>.
    Never falls back to getattr(wrapper, name) to avoid Gymnasium deprecation warnings.
    """
    # 1) base env first (no warnings)
    base = getattr(env, "unwrapped", env)
    if hasattr(base, name):
        return getattr(base, name)

    # 2) ask wrappers in a warning-free way
    try:
        val = env.get_wrapper_attr(name)
        if val is not None:
            return val
    except Exception:
        pass

    # 3) walk the chain, but still use get_wrapper_attr only
    cur = env
    for _ in range(16):
        try:
            val = cur.get_wrapper_attr(name)
            if val is not None:
                return val
        except Exception:
            pass
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break

    raise AttributeError(f"Attribute '{name}' not found on env or wrappers")

def _get_hull(e):
    # Wrapper-safe and future-proof: use our robust accessor
    return _get_attr(e, "hull")

def _get_hull_com_x(e) -> float:
    hull = _get_hull(e)
    # massData.center is in local coords
    return float(hull.massData.center[0])

def _set_hull_com_x(e, x: float, mass: float = 2.0, radius: float = 0.10, y_fixed: float = 0.0):
    """
    Shift COM by attaching a circular fixture at (x, y_fixed) in HULL LOCAL coords.
    Stores the created fixture as hull._com_payload so we can replace it next call.
    """
    hull = _get_hull(e)
    x = float(x); mass = float(mass); radius = float(radius); y_fixed = float(y_fixed)

    # remove previous payload if any
    if hasattr(hull, "_com_payload") and hull._com_payload is not None:
        try:
            hull.DestroyFixture(hull._com_payload)
        except Exception:
            pass
        hull._com_payload = None

    # add new payload only if mass > 0
    if mass > 0.0 and radius > 0.0:
        area = math.pi * (radius ** 2)
        density = mass / area
        hull._com_payload = hull.CreateCircleFixture(
            pos=(x, y_fixed), radius=radius, density=density,
            friction=0.0, restitution=0.0
        )

    # recompute mass/inertia/center
    hull.ResetMassData()
    return float(hull.massData.center[0])  # local COM x

class AttachPayload(gym.Wrapper):
    """
    Ensures a COM payload fixture exists on the hull.

    On reset:
      - Read current episode 'COM_X' from env.context (sampled by CARL).
      - Create/replace the circular fixture at that x with given mass/radius/y.
      - Remember this episode's pinned x.

    On step (optional):
      - If keep_center=True and there is NO COM_X dynamic updater, re-assert the
        *episode's* pinned x every step (prevents drift/flicker for static setups).
    """
    def __init__(
        self, env,
        *, mass: float = 5.0, radius: float = 0.40, y_fixed: float = 0.0,
        keep_center: bool = False, default_x0: float = 0.0
    ):
        super().__init__(env)
        self.mass = float(mass)
        self.radius = float(radius)
        self.y_fixed = float(y_fixed)
        self.keep_center = bool(keep_center)
        self.default_x0 = float(default_x0)

        self._has_dyn_comx: bool | None = None
        self._episode_pin_x: float = self.default_x0  # set at reset()

    def _refresh_dyn_flag(self) -> None:
        if self._has_dyn_comx is not None:
            return
        base = getattr(self.env, "unwrapped", self.env)
        fns = getattr(base, "feature_update_fns", None)
        if fns is None:
            try:
                fns = self.env.get_wrapper_attr("feature_update_fns")
            except Exception:
                fns = None
        self._has_dyn_comx = isinstance(fns, dict) and ("COM_X" in fns)

    def _read_com_x_from_context(self) -> float:
        # Try the standard CARL location
        ctx = None
        try:
            ctx = self.env.get_wrapper_attr("context")
        except Exception:
            pass
        if ctx is None:
            # Fallback to base env attribute if exposed
            base = getattr(self.env, "unwrapped", self.env)
            ctx = getattr(base, "context", None)

        if isinstance(ctx, dict) and ("COM_X" in ctx):
            try:
                return float(ctx["COM_X"])
            except Exception:
                pass
        return self.default_x0

    def reset(self, *args, **kwargs):
        # Let CARL sample the context first
        out = self.env.reset(*args, **kwargs)

        # Detect whether a dynamic COM_X updater exists
        self._has_dyn_comx = None
        self._refresh_dyn_flag()

        # Read the *sampled* COM_X for this episode and attach payload there
        self._episode_pin_x = self._read_com_x_from_context()
        _set_hull_com_x(
            self.env,
            x=self._episode_pin_x,
            mass=self.mass,
            radius=self.radius,
            y_fixed=self.y_fixed,
        )
        return out

    def step(self, action):
        result = self.env.step(action)

        # If static setup (no COM_X updater) and keep_center=True, re-assert pin
        self._refresh_dyn_flag()
        if self.keep_center and not self._has_dyn_comx:
            _set_hull_com_x(
                self.env,
                x=self._episode_pin_x,
                mass=self.mass,
                radius=self.radius,
                y_fixed=self.y_fixed,
            )
        return result


def walker_env_factory(
    contexts=None, render_mode=None, ctx_to_observe=None,
    payloaded=False, keep_center=False, payload_kwargs=None
):
    """
    Factory for CARL Walker with optional COM payload.
    - 'contexts' lets CARL sample COM_X per episode (we use it on reset).
    - 'keep_center': if True and NO COM_X updater, re-assert per-episode COM_X every step.
    """
    if contexts is None:
        from carl.envs import CARLBipedalWalker as walker
        env = walker(render_mode=render_mode)
    else:
        env = CARLBipedalWalker(
            contexts=contexts,
            render_mode=render_mode,
            obs_context_features=ctx_to_observe
        )

    env.render_mode = render_mode

    if payloaded:
        pk = payload_kwargs or {}
        env = AttachPayload(
            env,
            mass=pk.get("mass", 1.5),
            radius=pk.get("radius", 0.25),
            y_fixed=pk.get("y_fixed", 0.0),
            keep_center=keep_center,
            default_x0=0.0,   # only used if COM_X is missing in context
        )

    return env