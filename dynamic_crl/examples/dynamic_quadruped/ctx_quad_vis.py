# run_go2_carl.py
from __future__ import annotations
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../')
sys.path.append(project_root)

from dynamic_crl.src.dmc_envs.dc_envs.carl_dm_quadruped import CARLDmcQuadrupedEnv as QuadEnv
from dynamic_crl.src.dmc_envs.dmc_utils.saving_mj_run_gifs import record_gif_from_env

def make_ctx_quad_env(**kwargs):
    return QuadEnv(**kwargs)

def static_context_test():
    # 0. test friction changes - for default carl you need to set all 3 friction types (but only tangential has an effect)
    # 1. normal gravity, strong wind in x
    # 2. very low gravity, strong wind in y
    # 3. negative gravity, tiny wind in y
    ctx = {
        0: {"gravity": 9.811, 'wind_x': 1000, 'wind_y': 0., 'wind_z': 0., 'friction_tangential': 1.11, 'friction_rolling': 1.0, 'friction_torsional': 1.0},
        1: {'gravity': 9.812, 'wind_x': 1000., 'wind_y': 0., 'wind_z': 0., 'friction_tangential': 0.0001, 'friction_rolling': 1.0, 'friction_torsional': 1.0},
        2: {"gravity": 0.1, 'wind_x': 0.1, 'wind_y': 1000., 'wind_z': 0., 'friction_tangential': 1.0, 'friction_rolling': 1.0, 'friction_torsional': 1.0},
        3: {"gravity": -0.1, 'wind_x': -0.1, 'wind_y': 10., 'wind_z': 0., 'friction_tangential': 1.0, 'friction_rolling': 1.0, 'friction_torsional': 1.0},
    }
    env = make_ctx_quad_env(render_mode="human", contexts=ctx)
    # 1 - normal gravity, strong wind in x
    env.reset()
    env.render()
    # 2 - very low gravity, strong wind in y
    env.reset()
    env.render()
    # 3 - negative gravity, tiny wind in y
    env.reset()
    env.render()
    # 4 - normal gravity, strong wind in x, very low friction
    env.reset()
    env.render()

def test_no_ctx():
    ctx = {0: {}}
    env = make_ctx_quad_env(render_mode="human", contexts=ctx)
    env.reset()
    # call the wrapper's render (it handles viewer.launch internally)
    env.render()
    env.reset()
    env.render()
    # now really heavy payload
    env.reset()
    env.render()

###
def dynamic_context_test():

    # to test time limit, set it to 1s

    dynamic_fns_g = {
        "gravity": lambda step: -9.8 if step >= 50 else 0.1 ,  # switch gravity every 500 steps
        "d_wind": lambda step: [300, 0, 0] if step >= 100 else [0, 0, 0], 
        #"joint_damping": lambda step: 0.0 if step >= 150 else 0.1, # switch damping at t=150
        "friction": lambda step: 0.0001 if step >= 150 else 1.0, # switch friction at t=150
    }

    # default context
    ctx = {
        0: {"gravity": -2.5,
            },
        1: {"gravity": -2.5,
            "wind_x": 0.0, "wind_y": 0.0, "wind_z": 0.0,
          }
    }

    run_env = make_ctx_quad_env(
        render_mode="human",
        contexts=ctx,
        dc_updaters=dynamic_fns_g,
        task_kwargs={"time_limit": 10.0, "random": 10, "is_evaluation": True}, # very short time limit for testing
        obs_context_features=['gravity', 'wind_x', 'wind_y', 'wind_z'],
        quad_env_kwargs={'scale_toe_fric': True} # whether to scale toe friction along with floor friction
    )

    for ep in range(1):
        obs = run_env.reset()
        done = False
        for i in range(200):
            obs, reward, terminated, done, info = run_env.step(run_env.action_space.sample())
            print(f"step {i} - height: {run_env.env.env.physics.named.data.xipos['torso', 'z']} - gravity: {run_env.env.env.physics.model.opt.gravity[2]:.3f}")
            print(f"         x pose: {run_env.env.env.physics.named.data.xipos['torso', 'x']:.3f}, y pose: {run_env.env.env.physics.named.data.xpos['torso', 'y']:.3f}")
            if done or terminated:
                print(f"Episode finished after {i+1} timesteps for reason: {'terminated' if terminated else 'time limit reached'}")
                break

    env = make_ctx_quad_env(
        render_mode="human",
        contexts=ctx,
        dc_updaters=dynamic_fns_g,
        scale_toe_fric=True,  # whether to scale toe friction along with floor friction
        task_kwargs={"time_limit": 100.0, "random": 10, 'easy_init': True}, # very short time limit for testing
        obs_context_features=['gravity', 'wind_x', 'wind_y', 'wind_z'],
        quad_env_kwargs={'scale_toe_fric': True}
    )

    env.reset()
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
        env = make_ctx_quad_env(
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
        env = make_ctx_quad_env(
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

    # test_no_ctx()

    save_gif = False
    show_viewer = True

    if show_viewer:
        dynamic_context_test()

    if save_gif:
        dynamic_context_test_save_gif()
