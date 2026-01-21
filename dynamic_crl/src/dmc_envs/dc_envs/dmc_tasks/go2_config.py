ENV_CFG = dict(
    num_actions=12,
    default_joint_angles={
        # small ab/adduction to splay feet
        "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.00,
        # same thigh on all legs
        "FL_thigh_joint": 0.80, "FR_thigh_joint": 0.80, "RL_thigh_joint": 1.00, "RR_thigh_joint": 1.00,
        # same calf on all legs
        "FL_calf_joint": -1.50, "FR_calf_joint": -1.50, "RL_calf_joint": -1.50, "RR_calf_joint": -1.50,
    },
    kp=20.0, kd=0.5,
    base_init_pos=(0.0, 0.0, 0.42),   # spawn height
    base_init_quat=(1.0, 0.0, 0.0, 0.0),  # wxyz
    action_scale=0.25,                 # Δq scale for residual position control
    simulate_action_latency=True,
    clip_actions=100.0,  # not really used here; we clamp to [-1,1] before scaling
    resampling_time_s=4.0,
    term_pitch_deg=10.0,
    term_roll_deg=10.0,
)

OBS_CFG = dict(
    num_obs=48,
    obs_scales=dict(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05),
)

REWARD_CFG = dict(
    tracking_sigma=0.25,
    base_height_target=0.30,
    reward_scales=dict(
        tracking_lin_vel=1.0,
        tracking_ang_vel=0.2,
        lin_vel_z=-1.0,
        base_height=-50.0,
        action_rate=-0.005,
        similar_to_default=-0.1,
        # jump-related omitted here; easy to add back if you want it
    ),
)

COMMAND_CFG = dict(
    # [lin_vel_x, lin_vel_y, ang_vel, height, jump] (we will ignore jump here)
    num_commands=5,
    lin_vel_x_range=(-1.0, 2.0),
    lin_vel_y_range=(-0.5, 0.5),
    ang_vel_range=(-0.6, 0.6),
    height_range=(0.2, 0.4),
)