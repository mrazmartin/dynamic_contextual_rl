from carl.envs import CARLCartPole as cartpole
from carl.envs import CARLBipedalWalker as walker

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

from dynamic_crl.src.utils.log_msgs import warn_msg, info_msg

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

def walker_env_factory(contexts=None, render_mode=None, ctx_to_observe=None):
    """
    Factory function to create a CARL Walker environment with optional rendering and seed.
    """

    if contexts is None:
        env = walker(render_mode=render_mode)
        warn_msg("Not getting contexts from factory function, using default CARL Walker env.")
    else:
        env = walker(contexts=contexts, render_mode=render_mode, obs_context_features=ctx_to_observe)

    env.render_mode = render_mode

    return env