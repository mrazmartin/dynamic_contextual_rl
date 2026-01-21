# run_go2_carl.py
from __future__ import annotations
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../..')
sys.path.append(project_root)

from dynamic_crl.src.dmc_envs.dc_envs.go2_as_dmc import CARLDmcGo2Env as Go2Env
from dynamic_crl.src.dmc_envs.dmc_utils.saving_mj_run_gifs import record_gif_from_env

def make_ctx_go2_env(**kwargs):
    return Go2Env(**kwargs)

def static_context_test():
    ctx = {
        0: {"gravity_z": -10.0, "payload_mass": 1.0},
        1: {"gravity_z": -0.1, "payload_mass": 1.0},
        2: {"gravity_z": -0.1, "payload_mass": 10_000.0}
    }
    env = make_ctx_go2_env(render_mode="human", contexts=ctx)
    env.reset()
    # call the wrapper's render (it handles viewer.launch internally)
    env.render()
    env.reset()
    env.render()
    # now really heavy payload
    env.reset()
    env.render()

###
def dynamic_context_test(gravity: bool = True, mass: bool = True):

    dynamic_fns_g = {
        "gravity_z": lambda step: -9.8 if step >= 500 else -0.1 ,  # switch gravity every 500 steps
    }

    dynamic_fns_mass = {
        "payload_mass": lambda step: 5_000.0 if step >= 10 else 1.0 ,   # switch payload every 500 steps
    }

    # default context
    ctx = {0: {"gravity_z": -0.5, "payload_mass": 1.0},
          }

    if gravity:

        env = make_ctx_go2_env(
            render_mode="human",
            contexts=ctx,
            dc_updaters=dynamic_fns_g
        )
        for ep in range(1):
            obs = env.reset()
            done = False
            for i in range(500):
                obs, reward, done, terminated, info = env.step(env.action_space.sample())
                print(f"step {i} - height: {env.env.env.physics.base_height()}")

                # call the wrapper's render (it handles viewer.launch internally)
        env.reset()
        env.render()

    # now make it swing larger weight
    if mass:
        env = make_ctx_go2_env(
            render_mode="human",
            contexts=ctx,
            dc_updaters=dynamic_fns_mass
        )
        # doesnt do much since our dynamics are way later...
        for ep in range(1):
            obs = env.reset()
            done = False
            for i in range(500):
                obs, reward, done, terminated, info = env.step(env.action_space.sample())
                print(f"step {i} - height: {env.env.env.physics.base_height()}")
                # call the wrapper's render (it handles viewer.launch internally)

        env.reset()
        print("initial context gravity:", env.env.env.physics.model.opt.gravity)
        env.render()


def dynamic_context_test_save_gif(gravity: bool = True, mass: bool = True):
    ctx = {
        0: {"gravity_z": -0.5, "payload_mass": 1.0},
    }

    dynamic_fns_g = {
        "gravity_z": lambda step: -9.8 if step >= 500 else -0.1,   # switch gravity at t=600
        #"payload_mass": lambda step: 10000 if step >= 200 else 2.0,
    }

    dynamic_fns_mass = {
        "payload_mass": lambda step: 5_000 if step >= 10 else 1.0,
    }

    if gravity:
        env = make_ctx_go2_env(
            render_mode="rgb_array",
            contexts=ctx,
            dc_updaters=dynamic_fns_g
        )

        policy = env._random_policy()

        record_gif_from_env(
            env,
            out_path="_gifs/go2_dynamic_gravity.gif",
            steps=2000, # 4 seconds at 0.02s control step
            size=(480, 640),
            camera_id="track_static", # or "track" or "chase"
            policy_fn=policy,           # use a calmer policy than random if you want
            viewer_min_delay_ms=100,
        )

    if mass:
        env = make_ctx_go2_env(
            render_mode="rgb_array",
            contexts=ctx,
            dc_updaters=dynamic_fns_mass
        )

        policy = env._random_policy()

        record_gif_from_env(
            env,
            out_path="_gifs/go2_dynamic_mass.gif",
            steps=2000, # 4 seconds at 0.02s control step
            size=(480, 640),
            camera_id="track_static", # or "track" or "chase"
            policy_fn=policy,           # use a calmer policy than random if you want
            viewer_min_delay_ms=100,
        )

if __name__ == "__main__":
    # example context
    # static_context_test()

    save_gif = False
    show_viewer = True

    include_gravity = True
    include_mass = True

    if show_viewer:
        dynamic_context_test(gravity=include_gravity, mass=include_mass)

    if save_gif:
        dynamic_context_test_save_gif(gravity=include_gravity, mass=include_mass)
