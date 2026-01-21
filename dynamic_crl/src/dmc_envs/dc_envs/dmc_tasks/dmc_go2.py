# dmc_go2.py  (PD-free)
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import dm_env
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_env import specs

from dm_control import viewer

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../../..'))
from dynamic_crl.src.dmc_envs.context_utils import adapt_context_from_physics

# just so we can load all the tasks from this module...
from dm_control.utils import containers
SUITE = containers.TaggedTasks()

_ASSETS_DIR = Path(__file__).resolve().parent / ".." / "env_assets" / "unitree_go2"
_XML = _ASSETS_DIR / "scene_weight.xml"

# ---------- Physics helpers ----------
BASE_BODY = "base"
FEET = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
HIP_JOINTS   = ("FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint")
THIGH_JOINTS = ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint")
KNEE_JOINTS  = ("FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint")
GO2_JOINTS: tuple[str, ...] = HIP_JOINTS + THIGH_JOINTS + KNEE_JOINTS  # 12 DOF

class Physics(mujoco.Physics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scales = dict(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05)
        self._default_dof_pos: np.ndarray | None = None
        # self._free_joint_name: str | None = self._infer_free_joint_name() # can be removed?
        # self._go2_dof_idx = self._compute_go2_dof_idx()

        self._debug_prints = False  # set True to enable some debug prints

        # pd control
        self.control_type = "torque"  # "torque" or "pd"
        self.stiffness = 20.0 # Kp default for PD
        self.damping   = 0.5  # Kd default for PD
        self.action_scale = 0.25

        # cache end-effector (foot) body ids: use calf bodies in this model
        try:
            self._foot_body_ids = np.asarray(
                [self.model.name2id(n, mujoco.mjtObj.mjOBJ_BODY) for n in FEET],
                dtype=np.int32
            )
        except Exception:
            raise RuntimeError("Could not find foot bodies; did the model change?")
            self._foot_body_ids = np.array([], dtype=np.int32)

        # cache ground geom ids (by name and by type/owner fallback)
        self._ground_geom_ids = self._compute_ground_geom_ids()

    # --- setters without reset ---
    def set_joints_noreset(self, joint_dict: Dict[str, float], *, clip_to_range: bool = True) -> None:
        # WARNING: reset would undo the joint setting (DO NOT DELETE THIS COMMENT)
        for name, target in joint_dict.items():
            jid = self.model.name2id(name, mujoco.mjtObj.mjOBJ_JOINT)
            if clip_to_range:
                lo, hi = self.model.jnt_range[jid]
                target = float(np.clip(target, lo, hi))
            self.named.data.qpos[name] = float(target)
        self.forward()

    # can be removed?
    def set_base_z_noreset(self, z_abs: float) -> None:
        free = np.nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if free.size == 0:
            raise RuntimeError("No free joint; base is fixed.")
        jid  = int(free[0])
        qadr = int(self.model.jnt_qposadr[jid])  # [x,y,z,qw,qx,qy,qz]
        self.data.qpos[qadr + 2] = float(z_abs)
        self.forward()

    # --- free joint detection ---
    # can be removed? not used?
    def _infer_free_joint_name(self) -> str | None:
        free_ids = np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if free_ids.size == 0:
            return None
        jid = int(free_ids[0])
        return self.model.id2name(jid, mujoco.mjtObj.mjOBJ_JOINT)

    # --- cached observations ---
    # IMPORTANT: USED IN common_obs()
    def reset_default_dof_pos(self):
        self._default_dof_pos = self.named.data.qpos[list(GO2_JOINTS)].copy().astype(np.float64)

    def base_rotmat_body_to_world(self) -> np.ndarray:
        return self.named.data.xmat[BASE_BODY].reshape(3, 3)

    # IMPORTANT: keep this (used in obs)
    def base_lin_ang_vel_body(self) -> tuple[np.ndarray, np.ndarray]:
        c6 = self.named.data.cvel[BASE_BODY]
        ang_b = np.asarray(c6[:3], dtype=np.float64)
        lin_b = np.asarray(c6[3:6], dtype=np.float64)
        return lin_b, ang_b

    # IMPORTANT: keep this (used in obs)
    def gravity_body(self) -> np.ndarray:
        R_bw = self.base_rotmat_body_to_world()
        R_wb = R_bw.T
        g_w = self.model.opt.gravity.copy()
        g_norm = np.linalg.norm(g_w)
        g_hat = g_w / g_norm if g_norm > 0 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return R_wb @ g_hat

    # IMPORTANT: keep this (used in obs)
    def joint_pos_vel(self) -> tuple[np.ndarray, np.ndarray]:
        q  = self.named.data.qpos[list(GO2_JOINTS)].astype(np.float64)
        qd = self.named.data.qvel[list(GO2_JOINTS)].astype(np.float64)
        return q, qd

    def roll_pitch(self) -> tuple[float, float]:
        R = self.base_rotmat_body_to_world()
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        return float(roll), float(pitch)

    # probably can be removed?
    def _compute_go2_dof_idx(self) -> np.ndarray:
        idx = []
        for j in GO2_JOINTS:
            jid = self.model.name2id(j, mujoco.mjtObj.mjOBJ_JOINT)
            idx.append(int(self.model.jnt_dofadr[jid]))
        return np.asarray(idx, dtype=np.int32)

    # --- foot contact detection (for reward later) ---
    def _compute_ground_geom_ids(self) -> np.ndarray:
        """
        Try to identify ground geoms robustly:
          1) names containing 'floor'/'ground'/'groundplane'
          2) any geom that is a PLANE and belongs to 'world' body
        """
        ngeom = int(self.model.ngeom)
        if ngeom <= 0:
            return np.array([], dtype=np.int32)

        ground_ids = set()

        # (a) name-based
        KEYWORDS = ("floor", "ground", "groundplane")
        for gid in range(ngeom):
            nm = self.model.id2name(gid, mujoco.mjtObj.mjOBJ_GEOM)
            if isinstance(nm, bytes):
                nm = nm.decode("utf-8", errors="ignore")
            nm_l = (nm or "").lower()
            if any(k in nm_l for k in KEYWORDS):
                ground_ids.add(gid)

        # (b) fallback: plane on world
        try:
            world_bid = self.model.name2id("world", mujoco.mjtObj.mjOBJ_BODY)
        except Exception:
            world_bid = 0  # world is usually 0

        GEOM_PLANE = mujoco.mjtGeom.mjGEOM_PLANE
        for gid in range(ngeom):
            if int(self.model.geom_type[gid]) == GEOM_PLANE and int(self.model.geom_bodyid[gid]) == int(world_bid):
                ground_ids.add(gid)

        return np.asarray(sorted(ground_ids), dtype=np.int32)

    def feet_contact_mask(self, *, ground_only: bool = True) -> np.ndarray:
        """
        Boolean mask (len=4) for which 'feet' (calf bodies) are in contact.
        If ground_only=True, only count contacts with ground geoms.
        """
        nfeet = len(FEET)
        mask = np.zeros(nfeet, dtype=bool)
        if self._foot_body_ids.size != nfeet:
            return mask

        ncon = int(self.data.ncon)
        if ncon == 0:
            return mask

        geom_bodyid = self.model.geom_bodyid
        contacts = self.data.contact

        ground = set(int(g) for g in np.asarray(self._ground_geom_ids).tolist())
        for i in range(ncon):
            c = contacts[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            # enforce ground-only if requested
            if ground_only and (g1 not in ground and g2 not in ground):
                continue

            b1, b2 = int(geom_bodyid[g1]), int(geom_bodyid[g2])

            # mark any foot bodies involved
            for k, fid in enumerate(self._foot_body_ids):
                if b1 == fid or b2 == fid:
                    mask[k] = True

        return mask

    def num_feet_on_floor(self) -> int:
        return int(self.feet_contact_mask(ground_only=True).sum())

    # --- common observation assembly ---
    # IMPORTANT TO KEEP
    def common_obs(self, *,
                layout: str = "standup",             # ignore for now, choose large obs sets later
                residual_jpos: bool = True,
                use_cached_q0: bool = True,
                apply_scales: bool = True,
                include_lin_vel: bool = False        # off by default (standing doesn't need it)
                ) -> np.ndarray:
        """
        Minimal-noise observations for a sit->stand task.

        Features included (with rationale):
        - ang_b (3): base angular velocity in body frame; teaches damping of wobbles during stand-up.
        - g_b (3): gravity vector in body frame; smooth orientation cue (uprightness) without angle wrap issues.
        - q_res (12): joint positions residual to seated keyframe; centers the policy's search around the start pose.
        - qd (12): joint velocities; enables smooth, non-overshooting joint motions.
        - lin_b (3, optional): base linear velocity in body frame; usually noise for stand-up, useful for walking/recovery.

        By default, prev_action and lin_b are excluded to reduce noise and keep the task simple.
        """

        # --- Base kinematics ---
        lin_b, ang_b = self.base_lin_ang_vel_body()  # lin_b will be excluded unless include_lin_vel=True
        q, qd        = self.joint_pos_vel()

        # --- Residual joints around seated keyframe q0 (reduces search space drift) ---
        # Make sure _default_dof_pos is set to the SEATED keyframe right after reset().
        if residual_jpos and use_cached_q0:
            if self._default_dof_pos is None:
                self.reset_default_dof_pos()
            q_res = q - self._default_dof_pos
        else:
            q_res = q

        # --- Optional per-channel scaling hooks (keep neutral = 1.0) ---
        if apply_scales:
            ang_b = ang_b * float(self._scales.get("ang_vel", 1.0))
            q_res = q_res * float(self._scales.get("dof_pos", 1.0))
            qd    = qd    * float(self._scales.get("dof_vel", 1.0))

        # --- Orientation via gravity vector in body frame (preferred over roll/pitch) ---
        g_b = np.asarray(self.gravity_body(), dtype=np.float32).reshape(3)

        # --- Convert core features to contiguous float32 ---
        ang_b = np.asarray(ang_b, dtype=np.float32).reshape(3)
        q_res = np.asarray(q_res, dtype=np.float32).reshape(12)
        qd    = np.asarray(qd,    dtype=np.float32).reshape(12)

        # --- Assemble observation parts ---
        parts = [
            ang_b,        # 3  (base rotational speed)
            g_b,          # 3  (uprightness/orientation)
            q_res,        # 12 (where the joints are relative to sit)
            qd,           # 12 (how fast joints are moving)
        ]

        # Optional: linear velocity (useful for walking; off by default for stand-up)
        if include_lin_vel:
            parts.append(np.asarray(lin_b, dtype=np.float32).reshape(3))  # 3

        obs = np.concatenate(parts, axis=0).astype(np.float32)

        # --- Shape guardrails ---
        # Base (no prev_action, no lin_vel): 3 + 3 + 12 + 12 = 30
        base_dim = 30
        expected = base_dim \
                + (3  if include_lin_vel     else 0)
        assert obs.size == expected, f"[common_obs] got {obs.size} dims, expected {expected}."

        return obs

    # --- task helpers ---
    def base_height(self) -> float:
        return float(self.named.data.xpos[BASE_BODY][2])

    # used for reward
    def up_alignment(self) -> float:
        """
        Return cosine of angle between base z-axis and world z-axis (1.0 = upright).
        """
        R_bw = self.base_rotmat_body_to_world()
        return float(R_bw[2, 2])

    # used to early terminate before falling over
    def fell_over(self, *, min_height: float = 0.15, max_tilt_deg: float = 60.0) -> bool:
        if self.base_height() < float(min_height):
            return True
        roll, pitch = self.roll_pitch()
        lim = np.deg2rad(max_tilt_deg)
        
        # for debugging
        # result = (abs(roll) > lim) or (abs(pitch) > lim)
        # print(f"[fell_over] roll={roll:.3f}, pitch={pitch:.3f} -> {result}")

        return (abs(roll) > lim) or (abs(pitch) > lim)

    # --- keyframe reset (DONT TOUCH) ---
    def reset_to_keyframe(self, name: Optional[str] = None, index: Optional[int] = None):
        if (name is None) == (index is None):
            raise ValueError("Provide exactly one of name or index.")
        if name is not None:
            kid = self.model.name2id(name, mujoco.mjtObj.mjOBJ_KEY)
            if kid < 0:
                raise ValueError(f"No keyframe named {name!r}.")
        else:
            if not (0 <= int(index) < int(self.model.nkey)):
                raise IndexError(f"Keyframe index {index} out of range [0, {self.model.nkey-1}].")
            kid = int(index)
        from dm_control.mujoco.wrapper.mjbindings import mjlib
        with self.reset_context():
            mjlib.mj_resetDataKeyframe(self.model.ptr, self.data.ptr, kid)
            self.forward()

    def reset_to_last_keyframe(self):
        if int(self.model.nkey) == 0:
            raise RuntimeError("Model has no keyframes.")
        self.reset_to_keyframe(index=int(self.model.nkey) - 1)

    # --- DYNAMIC UPDATES (DONT TOUCH) ---
    def dynamic_update_gravity(self, new_gz: float) -> None:
        self.model.opt.gravity[2] = float(new_gz)
        self.forward()
        print(f"Updated gravity_z to {new_gz:.3f}")
    
    def dynamic_update_payload_mass(self, new_mass: float, body_name: str="payload", verbose=False) -> None:
        m_wrap, d_wrap = self.model, self.data
        M, D = m_wrap, d_wrap

        bid = m_wrap.name2id(body_name, mujoco.mjtObj.mjOBJ_BODY)

        old = float(M.body_mass[bid])
        if not (new_mass > 0.0 and old > 0.0):
            return
        s = float(new_mass) / old

        if s == 1.0:
            return

        if verbose:
            import copy
            old_mass = float(copy.deepcopy(M.body_mass[bid]))
            old_inertia = copy.deepcopy(M.body_inertia[bid])

        #with self.reset_context():
        M.body_mass[bid]    = float(new_mass)
        M.body_inertia[bid] = M.body_inertia[bid] * s

        if verbose:
            print(f"Changing payload mass:",
                  f"\n\tnew mass: {new_mass:.3f} (old mass: {old_mass:.3f})",
                  f"\n\tnew inertia: {M.body_inertia[bid]}) (old inertia: {old_inertia})")

        self.forward()

    # === PD UTILITIES (DONT TOUCH) ===
    def set_pd_gains(self, p=None, d=None):
        n = 12
        if not hasattr(self, "_p_gains"):
            self._p_gains = np.full(n, float(self.stiffness), dtype=np.float64)
        if not hasattr(self, "_d_gains"):
            self._d_gains = np.full(n, float(self.damping),   dtype=np.float64)
        if p is not None:
            p = np.asarray(p, dtype=np.float64).reshape(-1)
            self._p_gains = np.full(n, float(p.item()), dtype=np.float64) if p.size == 1 else p
        if d is not None:
            d = np.asarray(d, dtype=np.float64).reshape(-1)
            self._d_gains = np.full(n, float(d.item()), dtype=np.float64) if d.size == 1 else d

    def set_action_residual_scale(self, scale=None):
        n = 12
        if not hasattr(self, "_action_scale"):
            self._action_scale = np.full(n, float(self.action_scale), dtype=np.float64)
        if scale is not None:
            s = np.asarray(scale, dtype=np.float64).reshape(-1)
            self._action_scale = np.full(n, float(s.item()), dtype=np.float64) if s.size == 1 else s

    def pd_torque_from_action(self, action, *, use_cached_q0=True):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != 12:
            raise ValueError(f"PD expects 12-dim action, got {a.size}")
        # ensure defaults
        self.set_pd_gains()              # pulls from self.stiffness/damping
        self.set_action_residual_scale() # pulls from self.action_scale
        # reference pose q0
        if use_cached_q0 and (self._default_dof_pos is None):
            self.reset_default_dof_pos()
        q0 = self._default_dof_pos if (use_cached_q0 and self._default_dof_pos is not None) \
            else self.named.data.qpos[list(GO2_JOINTS)].astype(np.float64)
        # current state
        q, qd = self.joint_pos_vel()
        # desired pose (residual around q0)
        q_des = q0 + self._action_scale * a
        # clip desired to joint limits
        for i, j in enumerate(GO2_JOINTS):
            jid = self.model.name2id(j, mujoco.mjtObj.mjOBJ_JOINT)
            lo, hi = self.model.jnt_range[jid]
            q_des[i] = float(np.clip(q_des[i], lo, hi))
        # PD torque
        e   = q_des - q
        tau = self._p_gains * e - self._d_gains * qd
        # clip to actuator limits
        ctrl_rng = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        tau = np.clip(tau, ctrl_rng[:, 0], ctrl_rng[:, 1])
        return tau.astype(np.float32)

    def apply_action(self, action):
        """Single entry point: maps policy action -> actuator command according to control_type."""
        mode = str(getattr(self, "control_type", "torque")).lower()
        if mode == "pd":
            tau = self.pd_torque_from_action(action, use_cached_q0=True)
            if self._debug_prints:
                print(f"PD torques: {tau}") # debugging only
            self.data.ctrl[:] = tau
        else:  # 'torque' (direct)
            # no scaling/clipping
            self.data.ctrl[:] = action

    # success detection (for stand-up task)
    def base_ang_vel_world(self) -> np.ndarray:
        # convert ω_body -> ω_world
        w_b = self.named.data.cvel[BASE_BODY][:3].astype(np.float64)
        R_bw = self.base_rotmat_body_to_world()
        return (R_bw @ w_b).astype(np.float64)

    @property
    def dt(self) -> float:
        # sim timestep (seconds)
        return float(self.model.opt.timestep)

    @property
    def actuator_dim(self) -> int:
        return int(self.model.nu)

# -------------------- TASKS --------------------
# empty task (no reward, no termination)
class SitStill(base.Task):
    def initialize_episode(self, physics: Physics, random_state=None):
        physics.reset_to_last_keyframe()  # start from last keyframe in your XML
        physics.data.qvel[:] = 0.0
        physics.data.ctrl[:] = 0.0

    def before_step(self, action, physics: Physics):
        physics.data.ctrl[:] = 0.0  # ignore actions

    def get_observation(self, physics: Physics):
        return {"observations": physics.common_obs(layout="FS-tutor-48", residual_jpos=True)}

    def get_reward(self, physics: Physics):
        return 0.0

# StayStill task, do not move, just keep your current pose
class StayStill(base.Task):
    """
    Hold a given keyframe pose (default: 'stand') and remain still/upright.
    Works with torque or PD residual control.
    """
    def __init__(
        self,
        target_key: str = "stand",
        min_alive_height: float = 0.05,
        max_tilt_deg: float = 60.0,
        include_prev_action: bool = False,
        pose_norm: str = "l1",          # 'l1' or 'l2' for pose error
        upright_tol_deg: float = 6.0,   # success check
        success_hold_s: float = 0.25,   # how long to be within tol
    ):
        super().__init__()
        self.target_key = target_key
        self.min_alive_height = float(min_alive_height)
        self.max_tilt_deg = float(max_tilt_deg)
        self.pose_norm = str(pose_norm)
        self._include_prev_action = bool(include_prev_action)
        self._upright_tol = float(np.deg2rad(upright_tol_deg))
        self._success_hold_s = float(success_hold_s)

        self.q_target = None
        self.h_target = None

        # runtime
        self._succ_streak = 0
        self._succ_awarded = False
        self._terminated = False
        self._last_components: Dict[str, float] = {}

        # eval log snapshot (episode local)
        self._t_elapsed_s = 0.0
        self._sum_pose_err = 0.0
        self._sum_tau_rms = 0.0
        self._best_upright = 0.0
        self._n_steps = 0
        self._first_success_time_s = None
        self._eval_log = {
            "success": 0.0,
            "time_to_still_s": None,
            "avg_pose_err": None,
            "avg_action_rms": None,
            "best_upright_score": None,
        }

    # ---------- helpers ----------
    def _set_target_from_key(self):
        """
        Populate self.q_target (12-dim for GO2_JOINTS) and self.h_target from a keyframe.
        """
        tmp_physics = Physics.from_xml_path(str(_XML))
        try:
            tmp_physics.reset_to_keyframe(name=self.target_key)
            tmp_physics.forward()
        except Exception:
            raise RuntimeError(f"Could not reset to '{self.target_key}' keyframe; does it exist?")
        self.h_target = float(tmp_physics.base_height())
        self.q_target = tmp_physics.named.data.qpos[list(GO2_JOINTS)].copy().astype(np.float32)

    def _pose_error(self, q: np.ndarray, q_tgt: np.ndarray) -> float:
        q = np.asarray(q, np.float32).reshape(-1)
        t = np.asarray(q_tgt, np.float32).reshape(-1)
        assert q.shape == t.shape
        diff = q - t
        if self.pose_norm == "l2":
            return float(np.sqrt(np.mean(diff**2)))
        else:
            return float(np.mean(np.abs(diff)))

    # ---------- base.Task API ----------
    def initialize_episode(self, physics: Physics, random_state=None):
        self._terminated = False
        self._succ_awarded = False
        self._succ_streak = 0

        # set target pose if needed
        if (self.q_target is None) or (self.h_target is None):
            self._set_target_from_key()

        # reset to target keyframe (default 'stand')
        try:
            physics.reset_to_keyframe(name=self.target_key)
        except Exception:
            physics.reset_to_last_keyframe()

        physics.data.qvel[:] = 0.0
        physics.data.ctrl[:] = 0.0

        # PD mode setup (mirror your StandUp)
        ctr_mode = str(getattr(physics, "control_type", "torque")).lower()
        if ctr_mode == "pd":
            physics.reset_default_dof_pos()
            physics.set_pd_gains(physics.stiffness, physics.damping)
            physics.set_action_residual_scale(physics.action_scale)

        # action history for obs + smoothness term
        act_dim = physics.actuator_dim
        self._prev_action = np.zeros(act_dim, dtype=np.float32)
        self._curr_action = np.zeros(act_dim, dtype=np.float32)

        # step-wise logs
        self._t_elapsed_s = 0.0
        self._sum_pose_err = 0.0
        self._sum_tau_rms = 0.0
        self._best_upright = 0.0
        self._n_steps = 0
        self._first_success_time_s = None
        self._eval_log.update({
            "success": 0.0,
            "time_to_still_s": None,
            "avg_pose_err": None,
            "avg_action_rms": None,
            "best_upright_score": None,
        })

    def before_step(self, action, physics: Physics):
        # we allow actions (agent can use them to hold pose)
        self._prev_action = np.asarray(self._curr_action, dtype=np.float32).copy()
        self._curr_action = np.asarray(action, dtype=np.float32).reshape(-1)
        physics.apply_action(action)

    def get_observation(self, physics: Physics):
        observations = physics.common_obs(
            layout="standup",
            residual_jpos=True,
            use_cached_q0=True,
            apply_scales=True,
            include_lin_vel=False
        )

        # Replace height progress (not meaningful for “stay still”) with pose error to target.
        # IMPORTANT: use named access since q_target was built with named joints.
        q_curr = physics.named.data.qpos[list(GO2_JOINTS)].astype(np.float32)
        pose_err = np.float32(self._pose_error(q_curr, self.q_target))

        observations = np.concatenate(
            [observations, np.array([pose_err], dtype=np.float32)],
            axis=0
        )

        if self._include_prev_action:
            act_dim = physics.actuator_dim
            prev_a = np.asarray(self._prev_action, dtype=np.float32).reshape(-1)
            if prev_a.size < act_dim:
                raise ValueError(f"Expected prev_action size >= {act_dim}, got {prev_a.size}")
            observations = np.concatenate([observations, prev_a], axis=0)

        return {"observations": observations}

    def action_spec(self, physics: Physics):
        if getattr(physics, "control_type", "torque").lower() == "pd":
            n = 12
            lo = -np.ones(n, dtype=np.float32)
            hi =  np.ones(n, dtype=np.float32)
            return specs.BoundedArray(shape=(n,), dtype=np.float32, minimum=lo, maximum=hi)
        else:
            rng = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float32)
            lo, hi = rng[:, 0], rng[:, 1]
            return specs.BoundedArray(shape=(lo.size,), dtype=np.float32, minimum=lo, maximum=hi)

    def _q_qd_named(self, physics: Physics):
        q  = physics.named.data.qpos[list(GO2_JOINTS)].astype(np.float32)
        qd = physics.named.data.qvel[list(GO2_JOINTS)].astype(np.float32)
        return q, qd

    def get_reward(self, physics: Physics):
        # --- signals ---
        lin_b, ang_b = physics.base_lin_ang_vel_body()      # body-frame linear & angular vel (3,)
        vx, vy, vz = float(lin_b[0]), float(lin_b[1]), float(lin_b[2])
        wz = float(ang_b[2])                                 # yaw rate
        body_z = float(physics.base_height())

        # named joint slices to match q_target
        q  = physics.named.data.qpos[list(GO2_JOINTS)].astype(np.float32)
        qd = physics.named.data.qvel[list(GO2_JOINTS)].astype(np.float32)

        # action stats
        a_prev = np.asarray(self._prev_action, dtype=np.float32)
        a_curr = np.asarray(self._curr_action, dtype=np.float32)
        if a_prev.size < a_curr.size:
            a_prev = np.pad(a_prev, (0, a_curr.size - a_prev.size))

        tau = physics.data.qfrc_actuator.astype(np.float32)

        # --- tracking-style rewards (commands are zeros here) ---
        # configs (tweak-friendly)
        sigma_lin   = 0.25  # how quickly we punish XY drift   [m/s]^2 scale
        sigma_ang   = 0.5   # yaw-rate tolerance               [rad/s]^2 scale
        sigma_zvel  = 0.10  # vertical vel tolerance           [m/s]^2
        sigma_h     = 0.02  # height error tolerance           [m]^2
        scale_pose  = 0.25  # pose err scale (mean abs rad)

        # 1) Track zero XY velocity
        lin_vel_error = vx*vx + vy*vy
        r_track_lin = float(np.exp(- lin_vel_error / sigma_lin))

        # 2) Track zero yaw rate
        ang_vel_error = wz*wz
        r_track_ang = float(np.exp(- ang_vel_error / sigma_ang))

        # 3) Keep vertical velocity ~0
        r_lin_vel_z = float(np.exp(- (vz*vz) / sigma_zvel))

        # 4) Keep close to default/target joint pose
        pose_err = float(np.mean(np.abs(q - self.q_target)))   # mean |rad|
        r_similar_to_default = float(1.0 / (1.0 + (pose_err / scale_pose)))

        # 5) Keep base height near target
        dz = body_z - float(self.h_target)
        r_base_height = float(np.exp(- (dz*dz) / sigma_h))

        # 6) Penalize action rate (changes) and effort
        r_action_rate_pen = float(np.mean((a_curr - a_prev)**2))      # penalty
        r_energy_pen      = float(np.sum(tau**2))                      # penalty

        # 7) Small smoothness via joint velocity (optional)
        qd_rms = float(np.sqrt(np.mean(qd**2)))
        r_still_vel = float(np.exp(- (qd_rms*qd_rms)))  # dimensionless

        # --- weighting (simple & interpretable) ---
        w_track_lin  = 2.0
        w_track_ang  = 1.0
        w_zvel       = 0.5
        w_pose       = 3.0
        w_height     = 1.0
        w_still_vel  = 0.5
        w_act_rate   = 0.02   # penalties
        w_energy     = 0.001
        time_pen     = 0.0005

        reward = (
            w_track_lin * r_track_lin +
            w_track_ang * r_track_ang +
            w_zvel      * r_lin_vel_z +
            w_pose      * r_similar_to_default +
            w_height    * r_base_height +
            w_still_vel * r_still_vel
            - w_act_rate * r_action_rate_pen
            - w_energy   * r_energy_pen
            - time_pen
        )

        # log components (helps you see what's driving learning)
        self._last_components = {
            "r_track_lin": r_track_lin,
            "r_track_ang": r_track_ang,
            "r_zvel": r_lin_vel_z,
            "r_pose": r_similar_to_default,
            "pose_err": pose_err,
            "r_height": r_base_height,
            "dz": float(abs(dz)),
            "r_still_vel": r_still_vel,
            "qd_rms": qd_rms,
            "pen_action_rate": r_action_rate_pen,
            "pen_energy": r_energy_pen,
            "tau_rms": float(np.sqrt(np.mean(tau**2))),
            "r_total_raw": float(reward),
        }
        return float(reward)

    def should_terminate_episode(self, physics: Physics):
        return physics.fell_over(
            min_height=self.min_alive_height,
            max_tilt_deg=self.max_tilt_deg
        )

    def _succ_state_iter(self, physics: Physics):
        # success if: upright and near target pose and low motion
        roll, pitch = physics.roll_pitch()

        # --- use named access to match how q_target was built ---
        q_curr  = physics.named.data.qpos[list(GO2_JOINTS)].astype(np.float32)
        qd_curr = physics.named.data.qvel[list(GO2_JOINTS)].astype(np.float32)

        pose_err = self._pose_error(q_curr, self.q_target)
        qd_rms   = float(np.sqrt(np.mean(qd_curr**2)))

        upright_ok = (abs(roll) < self._upright_tol) and (abs(pitch) < self._upright_tol)
        near_pose  = (pose_err < 0.03)   # tweak as needed
        still_enuf = (qd_rms  < 0.5)     # tweak as needed

        success_now = upright_ok and near_pose and still_enuf

        required = int(self._success_hold_s / max(1e-6, physics.dt))
        self._succ_streak = self._succ_streak + 1 if success_now else 0
        if (self._succ_streak >= required) and not self._succ_awarded:
            self._succ_awarded = True

    def after_step(self, physics: Physics):
        # success tracker
        self._succ_state_iter(physics)

        # eval aggregates
        self._t_elapsed_s += physics.dt
        self._n_steps += 1

        # --- use named access to align with q_target ---
        q_curr  = physics.named.data.qpos[list(GO2_JOINTS)].astype(np.float32)
        qd_curr = physics.named.data.qvel[list(GO2_JOINTS)].astype(np.float32)
        pose_err = self._pose_error(q_curr, self.q_target)
        self._sum_pose_err += float(pose_err)

        tau = physics.data.qfrc_actuator.astype(np.float32)
        self._sum_tau_rms += float(np.sqrt(np.mean(tau**2)))

        roll, pitch = physics.roll_pitch()
        upright_now = float(np.exp(-10.0 * (roll**2 + pitch**2)))
        self._best_upright = max(self._best_upright, upright_now)

        if (self._first_success_time_s is None) and self.stay_success():
            self._first_success_time_s = float(self._t_elapsed_s)
            self._eval_log["success"] = 1.0
            self._eval_log["time_to_still_s"] = self._first_success_time_s

        steps = max(1, self._n_steps)
        self._eval_log["avg_pose_err"] = float(self._sum_pose_err / steps)
        self._eval_log["avg_action_rms"] = float(self._sum_tau_rms / steps)
        self._eval_log["best_upright_score"] = float(self._best_upright)

        super().after_step(physics)

    def stay_success(self) -> bool:
        return bool(self._succ_awarded)

    # control.Environment checks this to detect early terminations (e.g., flipped)
    def get_termination(self, physics: Physics):
        flipped = physics.fell_over(
            min_height=self.min_alive_height,
            max_tilt_deg=self.max_tilt_deg
        )
        self._terminated = flipped
        return 0 if self._terminated else None

# stand up task (rise from sit to stand and hold)
class StandUp(base.Task):
    """Rise from low pose to upright and hold it. Dense shaping with action penalty."""
    def __init__(
        self,
        start_key: str = "sit",
        target_height: float = 0.32,
        min_alive_height: float = 0.05,
        max_tilt_deg: float = 60.0,
        # obs_params
        include_prev_action: bool = False,
    ):
        super().__init__()
        self.start_key = start_key
        self.h_target = float(target_height)
        self.min_alive_height = float(min_alive_height)
        self.max_tilt_deg = float(max_tilt_deg)
        self._last_components: Dict[str, float] = {}

        self._set_target_pose()

        # obs params
        self._include_prev_action = bool(include_prev_action)
        self._set_start_height(start_key)  # sets h_start and z_denom (must come after set_target_pose)

        # runtime
        self._gave_success = False
        self._terminated = False

        # for evaluation logs:
        # --- eval summary (episode-local) ---
        self._t_elapsed_s = 0.0
        self._best_height = -np.inf
        self._first_success_time_s = None
        # public, read-only-ish snapshot for anyone to inspect (wrapper, debug prints, etc.)
        self._eval_log = {
            "success": 0.0,             # ever succeeded this episode (0/1)
            "best_height": None,        # max base z so far (float or None)
            "time_to_stand_s": None,    # seconds to first success (or None)
            "avg_height": None, "avg_tau_rms": None
        }
    
    def _set_start_height(self):
        """
        Set
            self.h_start
            self.z_denom
        for normalized height obs.
        """
        tmp_physics = Physics.from_xml_path(str(_XML))
        try:
            tmp_physics.reset_to_keyframe(name=self.start_key)
            tmp_physics.forward()
            self.h_start = float(tmp_physics.base_height())
            self.z_denom = max(1e-6, (self.h_target - self.h_start))
        except Exception:
            raise RuntimeError(f"Could not reset to '{self.start_key}' keyframe; does it exist?")
        
    def _set_start_height(self, start_key):
        tmp_physics = Physics.from_xml_path(str(_XML))
        try:
            tmp_physics.reset_to_keyframe(name=start_key)
            tmp_physics.forward()
            self.h_start = float(tmp_physics.base_height())
            self.z_denom = max(1e-6, (self.h_target - self.h_start))
        except Exception:
            raise RuntimeError(f"Could not reset to '{start_key}' keyframe; does it exist?")

    def _set_target_pose(self, target_key="stand", target_height=None):
        """
        Set
            self.h_target
            self.q_target
        to target height and joint pose, if target height not given, use the one from the pose.
        """
        tmp_physics = Physics.from_xml_path(str(_XML))

        if target_height is not None:
            self.h_target = float(target_height)
        else:
            # try to get target height from 'stand' keyframe if it exists
            try:
                tmp_physics.reset_to_keyframe(name=target_key)
                tmp_physics.forward()
                z_stand = float(tmp_physics.base_height())
                self.h_target = z_stand
                self.q_target = tmp_physics.named.data.qpos[list(GO2_JOINTS)].copy().astype(np.float32)
            except Exception:
                raise RuntimeError(f"Could not reset to '{target_key}' keyframe; does it exist?")

    def initialize_episode(self, physics: Physics, random_state=None):
        self._terminated = False
        self._gave_success = False

        # --- pose reset ---
        try:
            physics.reset_to_keyframe(name=self.start_key)
        except Exception:
            physics.reset_to_last_keyframe()
        physics.data.qvel[:] = 0.0
        physics.data.ctrl[:] = 0.0

        # reset action history (for obs)
        act_dim = physics.actuator_dim
        self._prev_action = np.zeros(act_dim, dtype=np.float32)
        self._curr_action = np.zeros(act_dim, dtype=np.float32)

        # success tracking
        self._succ_streak = 0
        self._succ_awarded = False

        # --- PD init (only if we're in PD mode) ---
        ctr_mode = str(getattr(physics, "control_type", "torque")).lower()
        if ctr_mode == "pd":
            # 1) cache the nominal joint pose we add residuals to
            physics.reset_default_dof_pos()
            # 2) set PD gains from physics attributes
            physics.set_pd_gains(physics.stiffness, physics.damping)
            # 3) set action residual scale from physics attribute
            physics.set_action_residual_scale(physics.action_scale)

        # init height default and denom for normalized height obs for stand-up
        self.h_start = float(physics.base_height())
        self.z_denom = max(1e-6, (self.h_target - self.h_start))

        # eval summary reset
        self._t_elapsed_s = 0.0
        self._best_height = -np.inf
        self._first_success_time_s = None
        self._sum_height = 0.0
        self._sum_tau_rms = 0.0
        self._n_steps = 0
        self._eval_log.update({
            "success": 0.0,
            "best_height": None,
            "time_to_stand_s": None,
            "avg_height": None,
            "avg_tau_rms": None
            })

    def before_step(self, action, physics: Physics):
        self._prev_action = np.asarray(self._curr_action, dtype=np.float32).copy()
        self._curr_action = np.asarray(action, dtype=np.float32).reshape(-1)
        physics.apply_action(action)

    def get_observation(self, physics: Physics):

        observations = physics.common_obs(
            layout="standup",
            residual_jpos=True,
            use_cached_q0=True,
            apply_scales=True,
            include_lin_vel=False      # not needed for easy standing up task
        )

        # Body-center height normalized to [0,1]
        body_z = float(physics.base_height())
        body_z_norm = (body_z - self.h_start) / self.z_denom
        body_z_norm = np.clip(body_z_norm, 0.0, 1.0).astype(np.float32)

        observations = np.concatenate([observations, np.array([body_z_norm], dtype=np.float32)], axis=0)

        if self._include_prev_action:
            act_dim = physics.actuator_dim
            prev_a = np.asarray(self._prev_action, dtype=np.float32).reshape(-1)
            if prev_a.size < act_dim:
                raise ValueError(f"Expected prev_action size >= {act_dim}, got {prev_a.size}")
            observations = np.concatenate([observations, prev_a], axis=0)

        return {"observations": observations}

    # --- action specification ---
    def action_spec(self, physics: Physics):
        if getattr(physics, "control_type", "torque").lower() == "pd":
            """
            In PD mode, the action is a residual around the nominal joint positions.
            """
            n = 12
            lo = -np.ones(n, dtype=np.float32)
            hi =  np.ones(n, dtype=np.float32)
            return specs.BoundedArray(shape=(n,), dtype=np.float32, minimum=lo, maximum=hi)
        else:
            rng = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float32)
            lo, hi = rng[:, 0], rng[:, 1]
            return specs.BoundedArray(shape=(lo.size,), dtype=np.float32, minimum=lo, maximum=hi)

    # reward components
    def _pose_progress(self,
                    q: np.ndarray,
                    q_target: np.ndarray = None,
                    q_prev: np.ndarray | None = None,
                    norm: str = "l1",
                    deadzone: float = 1e-5,
                    positive_only: bool = True):
        """
        Reward = decrease in absolute distance to target between t-1 and t:
            r = max(0, d_prev - d_curr)
        Distance uses mean per-joint (scale-invariant across DOFs).
        - norm: "l1" -> mean |q - q_target| ; "l2" -> sqrt(mean (q - q_target)^2)
        - If q_prev is None, uses cached previous pose/distance; otherwise initializes lazily (r=0 on first call).
        """
        if q_target is None:
            q_target = self.q_target  # assume you set this elsewhere

        q      = np.asarray(q,        np.float32).reshape(-1)
        q_tgt  = np.asarray(q_target, np.float32).reshape(-1)
        assert q.shape == q_tgt.shape, "q and q_target must have same shape"

        # distance function
        diff = q - q_tgt
        if norm == "l2":
            d_curr = float(np.sqrt(np.mean(diff**2)))
        else:  # "l1" (default)
            d_curr = float(np.mean(np.abs(diff)))

        # previous distance: prefer explicit q_prev; else cached; else init to current
        if q_prev is not None:
            q_prev = np.asarray(q_prev, np.float32).reshape(-1)
            assert q_prev.shape == q.shape
            diff_prev = q_prev - q_tgt
            d_prev = float(np.sqrt(np.mean(diff_prev**2))) if norm == "l2" else float(np.mean(np.abs(diff_prev)))
        elif hasattr(self, "_pose_prev_d"):
            d_prev = float(self._pose_prev_d)
        else:
            d_prev = d_curr  # first step → no progress

        # progress reward (only pay for improvement)
        progress = d_prev - d_curr
        if abs(progress) < deadzone:
            progress = 0.0
        if positive_only and progress < 0.0:
            progress = 0.0

        # cache for next call (no need to touch initialize_episode)
        self._pose_prev_d = d_curr
        self._pose_prev_q = q.copy()

        info = {"pose_d_curr": d_curr, "pose_d_prev": d_prev, "pose_progress": progress}
        return float(progress), info
    
    def _feet_contact_reward(self, physics,
                            min_height_frac: float = 0.0,   # 0.0 = always active; e.g. 0.05 starts after 5% of stand height
                            ema_tau: float | None = None):  # None = no smoothing; else seconds for EMA
        """
        Returns r_feet in [0,1] = (# feet on ground) / 4.
        - min_height_frac: gate so you only reward feet contact once you're above the seat a bit.
        - ema_tau: optional smoothing (seconds). Set to None for raw, instant signal.
        """
        # height gate (optional)
        if min_height_frac > 0.0:
            gate_h = self.h_start + min_height_frac * self.z_denom
            if float(physics.base_height()) < gate_h:
                # reset smoothing if we gate
                if hasattr(self, "_feet_ema"): delattr(self, "_feet_ema")
                return 0.0, {"feet_frac": 0.0, "feet_count": 0}

        # raw contact fraction
        mask = physics.feet_contact_mask(ground_only=True).astype(np.float32)  # shape (4,), values 0/1
        frac = float(np.mean(mask))                                            # 0..1

        # optional smoothing
        if ema_tau is None:
            r = frac
        else:
            beta = min(1.0, physics.dt / max(physics.dt, float(ema_tau)))
            prev = getattr(self, "_feet_ema", frac)
            self._feet_ema = (1.0 - beta) * prev + beta * frac
            r = float(self._feet_ema)

        return r, {"feet_frac": frac, "feet_count": int(mask.sum())}

    # --- reward assembly ---
    def get_reward(self, physics: Physics):
        roll, pitch = physics.roll_pitch()
        ang_b = physics.base_lin_ang_vel_body()[1]
        body_z = float(physics.base_height())

        # progress in [0,1]
        height_prog = float(np.clip((body_z - self.h_start) / self.z_denom, 0.0, 1.0))

        # NEW: upward velocity reward (only positive lift)
        if getattr(self, "_prev_body_z", None) is None:
            dh = 0.0
        else:
            dh = (body_z - self._prev_body_z) / max(1e-6, physics.dt)
        self._prev_body_z = body_z
        r_lift = max(0.0, dh)  # reward only when COM goes up

        # terms
        r_upright = float(np.exp(-10.0 * (roll**2 + pitch**2)))
        r_height  = height_prog
        r_stab    = float(np.exp(-2.0 * np.dot(ang_b, ang_b)))

        a_prev = np.asarray(self._prev_action, dtype=np.float32)
        a_curr = np.asarray(self._curr_action,  dtype=np.float32)
        if a_prev.size < a_curr.size: a_prev = np.pad(a_prev, (0, a_curr.size - a_prev.size))
        r_smooth = - float(np.mean((a_curr - a_prev)**2))

        tau = physics.data.qfrc_actuator.astype(np.float32)
        r_energy = - 1e-4 * float(np.sum(tau**2))

        # weights: push height & lift harder; ease stability early
        w_upright, w_height, w_lift = 1.0, 4.0, 0.5
        w_stab, w_smooth, w_energy  = 0.2, 0.02, 0.001
        time_pen = 0.001  # per step; keep simple

        q, qd = physics.joint_pos_vel()

        r_pose_prog, pose_info = self._pose_progress(q, q_target=self.q_target, norm="l1", deadzone=1e-5)
        r_feet, feet_info = self._feet_contact_reward(
            physics,
            min_height_frac=0.1,   # set to 0.05 if you only want it once you're lifting
            ema_tau=None           # or e.g. 0.05 to smooth flicker
        )

        reward = (
            2.0 * r_pose_prog +
            0.5 * r_feet +
            w_upright*r_upright +
            w_height*r_height +
            w_lift*r_lift +
            w_stab*r_stab +
            w_smooth*r_smooth +
            w_energy*r_energy -
            time_pen)

        # Log for Monitor/overlay
        self._last_components = {
            "r_upright": r_upright, "r_height": r_height, "r_stab": r_stab,
            "r_smooth": r_smooth, "r_energy": r_energy, "r_total_raw": reward,
            "body_z": body_z, "height_prog": height_prog,
            "roll_deg": float(np.rad2deg(roll)), "pitch_deg": float(np.rad2deg(pitch)),
            #"ang_w_norm": float(np.linalg.norm(ang_w)),
            "tau_rms": float(np.sqrt(np.mean(tau**2))),
        }
        return float(reward)

    def should_terminate_episode(self, physics: Physics):
        return physics.fell_over(
            min_height=self.min_alive_height,
            max_tilt_deg=self.max_tilt_deg
        )
    
    def _succ_state_iter(self, physics: Physics):
        roll, pitch = physics.roll_pitch()
        ang_w = physics.base_ang_vel_world()
        body_z = float(physics.base_height())

        upright_tol = np.deg2rad(6.0)
        stable_now = (float(np.linalg.norm(ang_w)) < 0.4)
        success_now = (body_z >= self.h_target - 0.01) and \
                    (abs(roll) < upright_tol) and (abs(pitch) < upright_tol) and \
                    stable_now

        dt = physics.dt
        required = int(0.25 / max(1e-6, dt))
        self._succ_streak = self._succ_streak + 1 if success_now else 0
        if (self._succ_streak >= required) and not self._succ_awarded:
            self._succ_awarded = True

    def after_step(self, physics: Physics):
        # A. update success streak
        self._succ_state_iter(physics)

        # B. update eval summary
        self._t_elapsed_s += physics.dt
        z_now = float(physics.base_height())
        self._n_steps += 1
        self._sum_height += z_now
        self._best_height = max(self._best_height, z_now)

        tau = physics.data.qfrc_actuator.astype(np.float32)
        self._sum_tau_rms += float(np.sqrt(np.mean(tau**2)))

        if (self._first_success_time_s is None) and self.stand_success():
            self._first_success_time_s = float(self._t_elapsed_s)
            self._eval_log["success"] = 1.0
            self._eval_log["time_to_stand_s"] = self._first_success_time_s


        steps = max(1, self._n_steps)
        self._eval_log["avg_height"] = float(self._sum_height / steps)
        self._eval_log["avg_action_rms"] = float(self._sum_tau_rms / steps)
        self._eval_log["best_height"] = float(self._best_height)

        # C. always call parent (it drives the viewer reward overlay, etc.)
        super().after_step(physics)
    
    def stand_success(self):
        return bool(self._succ_awarded)

    # important for termination of flipping upside down
    # used by the control.Environment step() method to notice termination
    def get_termination(self, physics: Physics):
        flipped = physics.fell_over(
            min_height=self.min_alive_height,
            max_tilt_deg=self.max_tilt_deg
        )
        self._terminated = flipped

        return 0 if self._terminated else None

# ---------- Env factory ----------
def make_env(xml_path: str | Path = _XML, control_timestep: float = 0.02,
             time_limit: float = 10.0, task: Optional[base.Task] = None) -> control.Environment:
    physics = Physics.from_xml_path(str(xml_path))
    task = SitStill() if task is None else task
    return control.Environment(physics=physics, task=task,
                               control_timestep=control_timestep, time_limit=time_limit)

# used in test trainer ?? does anyone use this?????
def make_standup_env(start_key="sit", min_alive_height=0.05, max_tilt_deg=60.0, time_limit=10.0) -> control.Environment:
    env = make_env(
        task=StandUp(
            start_key=start_key,
            min_alive_height=min_alive_height,
            max_tilt_deg=max_tilt_deg,
            time_limit=time_limit)
    )
    return env

# ---------- Register with dm_control.suite ----------
@SUITE.add("stand_up")
def stand_up(*, context=None, environment_kwargs=None,
             control_type: str = "torque",
             stiffness: float = 20.0, # defaults from unitree rl
             damping: float = 0.5,  # defaults from unitree rl
             action_scale: float = 0.25, # defaults from unitree rl
             **task_kwargs):
    physics = Physics.from_xml_path(str(_XML))
    # configure control
    physics.control_type = control_type
    physics.stiffness    = float(stiffness)
    physics.damping      = float(damping)
    physics.action_scale = float(action_scale)

    task = StandUp(**task_kwargs)

    adapt_context_from_physics(physics, context)

    return control.Environment(
        physics=physics,
        task=task,
        **(environment_kwargs or {})
    )

@SUITE.add("stay_still")
def stay_still(*, context=None, environment_kwargs=None,
               control_type: str = "torque",
               stiffness: float = 20.0,
               damping: float = 0.5,
               action_scale: float = 0.25,
               **task_kwargs):
    """
    Hold a keyframe pose (default 'stand'). Accepts the same control args as stand_up.
    Pass task_kwargs like target_key="stand", min_alive_height=..., max_tilt_deg=...
    """
    physics = Physics.from_xml_path(str(_XML))

    # configure control (mirror stand_up)
    physics.control_type = control_type
    physics.stiffness    = float(stiffness)
    physics.damping      = float(damping)
    physics.action_scale = float(action_scale)

    task = StayStill(**task_kwargs)

    adapt_context_from_physics(physics, context)

    return control.Environment(
        physics=physics,
        task=task,
        **(environment_kwargs or {})
    )

# #################################################
# -------------------- TESTS -------------------- #
# #################################################

# ---------- Minimal test (zero & random policies) ----------
def test_go2_physics(
    seed: int = 0,
    steps: int = 5,
    use_stand_up: bool = True,
    verbose: bool = False,
    control_type: str = "pd",          # "pd" or "torque"
    stiffness: float = 20.0,
    damping: float = 0.5,
    action_scale: float = 0.25,
) -> bool:
    if control_type.lower() == "pd":
        env = _build_env_for_tests(
            start_key="sit",
            time_limit=2.0,
            control_type="pd",
            stiffness=stiffness,
            damping=damping,
            action_scale=action_scale,
        )
    else:
        # torque mode uses your simple make_env
        task = StandUp(start_key="sit") if use_stand_up else SitStill()
        env = make_env(task=task)

    ts = env.reset()
    physics: Physics = env.physics  # type: ignore

    if verbose:
        print("=== Action spec ===")
    a_spec = env.action_spec()
    if verbose:
        print(f"shape={a_spec.shape}, dtype={a_spec.dtype}, min={a_spec.minimum[:4]}, max={a_spec.maximum[:4]}")

    a_zero = np.zeros(a_spec.shape, dtype=a_spec.dtype)
    rng = np.random.default_rng(seed)
    a_rand = rng.uniform(a_spec.minimum, a_spec.maximum, size=a_spec.shape).astype(a_spec.dtype)

    if verbose:
        print("\n=== Physics helpers at reset ===")
        print(f"base_height()      = {physics.base_height():.4f}")
        print(f"up_alignment()     = {physics.up_alignment():.4f}")
        roll, pitch = physics.roll_pitch()
        print(f"roll, pitch (rad)  = {roll:.4f}, {pitch:.4f}")
        if control_type.lower() == "pd":
            _pd_authority_report(physics)

    # obs checks
    obs_48 = physics.common_obs(layout="FS-tutor-48", residual_jpos=True)
    if verbose:
        print("\n=== Observation shapes ===")
        print(f"obs_48 (FS-tutor-48):      {obs_48.shape}  # expect (48,)")

    if verbose:
        print("\n=== Stepping with zero action ===")
    for t in range(steps):
        ts = env.step(a_zero)
        if verbose:
            print(f"t={t:02d}: reward={ts.reward:.4f}, terminated={ts.last()}")

    if verbose:
        print("\n=== Stepping with random action ===")
    for t in range(steps):
        ts = env.step(a_rand)
        bh = physics.base_height()
        up = physics.up_alignment()
        roll, pitch = physics.roll_pitch()
        if verbose:
            print(f"t={t:02d}: reward={ts.reward:.4f}, height={bh:.3f}, up={up:.3f}, roll={roll:.3f}, pitch={pitch:.3f}")

    if verbose:
        print("\nOK ✓ physics helpers and obs generation exercised.")
    return True

# ---------- Testing env resets ----------
def _pose_summary(physics: Physics, tag: str, verbose: bool = True):
    """
    Return (base_height, up_alignment, roll, pitch) tuple and print if verbose.
    """
    bh = physics.base_height()
    up = physics.up_alignment()
    roll, pitch = physics.roll_pitch()
    if verbose:
        print(f"[{tag}] base_height={bh:.3f}, up={up:.3f}, roll={roll:.3f}, pitch={pitch:.3f}")
    return bh, up, roll, pitch

def test_resets(
    *,
    task: Optional[base.Task] = None,
    time_per_episode: float = 1.0,
    num_episodes: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> bool:
    """Run random policy, reset, and verify we return to the initial pose each episode."""
    env = make_env(task=task or StandUp(start_key="sit"), time_limit=time_per_episode)
    rng = np.random.default_rng(seed)
    a_spec = env.action_spec()

    # Episode 0 reset (initial)
    ts = env.reset()
    physics: Physics = env.physics  # type: ignore
    if verbose:
        print("\n=== After initial reset ===")
    control_values = _pose_summary(physics, "reset[0]", verbose) # pose after initial reset

    for ep in range(num_episodes):
        if verbose:
            print(f"\n--- Episode {ep} rollout ({time_per_episode} seconds, random policy) ---")
        while not ts.last():
            # random action
            a = rng.uniform(a_spec.minimum, a_spec.maximum, size=a_spec.shape).astype(a_spec.dtype)
            ts = env.step(a)
            if ts.last():
                print(f"Episode ended at {env._step_count * env.control_timestep()} / {time_per_episode} seconds.")
                # Time limit or termination — break early
                # print position at the end of the episode
                final_pose = _pose_summary(physics, f"end_ep{ep}", verbose)
                print(f"Final pose vs initial: {final_pose} vs {control_values}\n")
                break

        # Reset environment to initial pose/state for the *same task*
        ts = env.reset()

        after_reset = _pose_summary(physics, f"reset[{ep+1}]", verbose)
        if after_reset != control_values:
            if verbose:
                print(f"✗ MISMATCH after reset[{ep+1}] vs initial reset[0]")
            return False

    if verbose:
        print("\nOK ✓ resets return to the task's initial pose each episode.")
    return True

# ----- testing pd ------
# ---------- Small helpers for PD sanity ----------
def _build_env_for_tests(
    *,
    start_key: str = "sit",
    time_limit: float = 2.0,
    control_type: str = "pd",        # "pd" or "torque"
    stiffness: float = 20.0,
    damping: float = 0.5,
    action_scale: float = 0.25,
):
    """Create env via SUITE factory so control settings are applied consistently."""
    return stand_up(
        context=None,
        environment_kwargs=dict(time_limit=time_limit),
        control_type=control_type,
        stiffness=stiffness,
        damping=damping,
        action_scale=action_scale,
        start_key=start_key,
        min_alive_height=0.05,
        max_tilt_deg=60.0,
    )

def _pd_authority_report(physics: Physics):
    rng = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float64)
    tau_max_med = float(np.median(rng[:, 1]))
    Kp = float(getattr(physics, "stiffness", 20.0))
    scale = float(getattr(physics, "action_scale", 0.25))
    print(f"[PD AUTH] Kp*scale = {Kp*scale:.3f}   median(tau_max) ≈ {tau_max_med:.3f}")
    return Kp * scale, tau_max_med

def test_pd_sanity(
    *,
    start_key: str = "sit",
    time_limit: float = 1.0,
    stiffness: float = 30.0,
    damping: float = 1.0,
    action_scale: float = 0.35,
    seed: int = 0,
    steps: int = 10,
) -> bool:
    """Sanity check PD path:
       - action_spec is [-1,1]
       - random residual action yields torques within ctrlrange
       - report saturation fraction and authority
    """
    env = _build_env_for_tests(
        start_key=start_key,
        time_limit=time_limit,
        control_type="pd",
        stiffness=stiffness,
        damping=damping,
        action_scale=action_scale,
    )
    ts = env.reset()
    physics: Physics = env.physics  # type: ignore

    print("\n=== PD sanity ===")
    print("control_type:", physics.control_type)

    a_spec = env.action_spec()
    print("action_spec min/max (first 4):", a_spec.minimum[:4], a_spec.maximum[:4])
    assert np.allclose(a_spec.minimum, -1.0), "PD action_spec minimum must be -1"
    assert np.allclose(a_spec.maximum,  1.0), "PD action_spec maximum must be +1"

    # Authority report
    Kp_scale, tau_med = _pd_authority_report(physics)

    # Random residuals in [-1,1]
    rng = np.random.default_rng(seed)
    sat_count = 0
    tot = 0

    # Precompute ctrlrange for saturation statistics
    ctrl_rng = np.asarray(physics.model.actuator_ctrlrange, dtype=np.float64)
    lo, hi = ctrl_rng[:, 0], ctrl_rng[:, 1]

    for t in range(steps):
        a = rng.uniform(-1.0, 1.0, size=a_spec.shape).astype(a_spec.dtype)
        # compute PD torques (defensive clamp happens inside)
        tau = physics.pd_torque_from_action(a, use_cached_q0=True).astype(np.float64)
        # check in-range
        assert np.all(tau >= lo - 1e-6) and np.all(tau <= hi + 1e-6), "PD torque out of ctrlrange"
        # push into sim for a real step
        ts = env.step(a)

        # saturation stats
        sat = np.logical_or(tau <= (lo + 1e-6), tau >= (hi - 1e-6))
        sat_count += int(np.sum(sat))
        tot += tau.size

        if t < 3:
            print(f"t={t:02d}  ||tau||_inf={np.max(np.abs(tau)):.3f}  height={physics.base_height():.3f}  up={physics.up_alignment():.3f}  reward={float(ts.reward):.3f}")

    sat_frac = sat_count / max(1, tot)
    print(f"Saturation fraction over {steps} steps: {sat_frac*100:.1f}%")
    print("OK ✓ PD path produces bounded torques.")
    return True

# ---------- Viewer wiring (zero or random policy) ----------
def zero_policy(env: control.Environment):
    a_spec = env.action_spec()
    zero = np.zeros(a_spec.shape, dtype=a_spec.dtype)
    return lambda _ts: zero

def random_policy(env: control.Environment, seed: int = 0):
    a_spec = env.action_spec()
    rng = np.random.default_rng(seed)
    return lambda _ts: rng.uniform(a_spec.minimum, a_spec.maximum, size=a_spec.shape).astype(a_spec.dtype)

if __name__ == "__main__":

    test_physics = True
    test_env_resets = True

    test_stand_up = True
    start_key = "sit"  # "sit" / "tiptoe" / "home"

    if test_physics:
        ok = test_go2_physics(
            use_stand_up=test_stand_up,
            verbose=True,
            control_type="pd",      # try "torque" to compare
            stiffness=30.0,
            damping=1.0,
            action_scale=0.35,
        )
        if not ok:
            raise ValueError("test_go2_physics failed.")

        ok = test_pd_sanity(
            start_key="sit",
            time_limit=1.0,
            stiffness=30.0,
            damping=1.0,
            action_scale=0.35,
            steps=10,
        )
        if not ok:
            raise ValueError("test_pd_sanity failed.")

    if test_env_resets:
        test_resets(
            task=StandUp(
                start_key=start_key,
                min_alive_height=0.05,
                max_tilt_deg=60.0,
            ),
            time_per_episode=5.0,
            num_episodes=2,
            seed=42,
            verbose=True,
        )

    # Viewer path (through factory so PD config applies)
    if True:
        env = _build_env_for_tests(
            start_key=start_key,
            time_limit=float("inf"),
            control_type="pd",
            stiffness=30.0,
            damping=1.0,
            action_scale=0.35,
        )
        viewer.launch(env, policy=random_policy(env, seed=123))
