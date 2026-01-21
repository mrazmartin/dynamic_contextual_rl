#!/usr/bin/env python3
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# --- project path (adjust if needed) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../..")
sys.path.append(project_root)

from dynamic_crl.src.dmc_envs.dc_envs.carl_dm_quadruped import CARLDmcQuadrupedEnv

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

    print("Showing dynamic contexts - wind/friction")
    print("Exhibit [1] - low gravity")
    env.render()
    print("Exhibit [2] - standard gravity")
    env.reset()
    env.render()