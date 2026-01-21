import imageio
import numpy as np
import cv2
import dm_env
import os, sys

# Ensure project root on path (adjust levels if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../../..')
sys.path.append(project_root)
from dynamic_crl.src.dmc_envs.dc_envs.go2_as_dmc import _ViewerHookEnv


def annotate_with_time(frame: np.ndarray, sim_time: float) -> np.ndarray:
    fb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    text = f"t = {sim_time:.2f} s"
    h, w, _ = fb.shape
    cv2.putText(fb, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)


def _pick_world_camera_id(physics) -> int | None:
    """Prefer a camera attached to the worldbody (stable)."""
    m = physics.model
    n = int(getattr(m, "ncam", 0))
    if n <= 0:
        return None
    for i in range(n):
        if int(m.cam_bodyid[i]) == 0:  # 0 == world
            return i
    return 0  # fallback: first camera


def record_gif_from_env(
    go2_env,
    out_path: str,
    steps: int = 2000,
    size: tuple[int, int] = (480, 640),
    camera_id: int | str | None = "track_static",
    policy_fn=None,
    slowmo: float = 1.0,
    # >>> new: assume many viewers clamp to ~40 ms; tune if needed
    viewer_min_delay_ms: int = 40,
    stop_on_first_episode: bool = True,
    include_terminal_frame: bool = False,
):
    """
    Encodes GIF with per-frame durations derived from MuJoCo sim time,
    quantized to the viewer's minimum frame delay to avoid slow playback.
    """
    import imageio, numpy as np, dm_env, cv2

    if slowmo < 1.0:
        slowmo = 1.0

    go2_env.reset()
    hooked = _ViewerHookEnv(go2_env)
    if policy_fn is None:
        policy_fn = go2_env._viewer_policy or go2_env._random_policy()

    ts: dm_env.TimeStep = hooked.reset()
    frames: list[np.ndarray] = []
    durations: list[float] = []

    # Auto-pick world-fixed camera if None
    if camera_id is None:
        maybe_cam = _pick_world_camera_id(hooked.physics)
        if maybe_cam is not None:
            camera_id = int(maybe_cam)
            print(f"[camera] Using fixed world camera id={camera_id}")
        else:
            print("[camera] No world-attached cameras found; view may move.")

    # --- quantization setup ---
    q = max(0.01, viewer_min_delay_ms / 1000.0)  # GIF delay quantum in seconds (10/20/40ms common)
    min_keep_dt = q / slowmo                      # required sim-time gap to keep a frame

    def quantize_duration(x: float) -> float:
        # round to nearest multiple of q, and clamp to at least q
        return max(q, round(x / q) * q)

    def grab_frame():
        h, w = size
        return hooked.physics.render(height=h, width=w, camera_id=camera_id)

    # first frame
    last_kept_time = hooked.physics.data.time
    frames.append(annotate_with_time(grab_frame(), last_kept_time))
    durations.append(q)  # first frame shown for exactly one quantum

    t = 0
    while t < steps:
        action = policy_fn(ts)
        ts = hooked.step(action)
        sim_time = hooked.physics.data.time

        if (sim_time - last_kept_time) >= min_keep_dt:
            frame = grab_frame()
            if isinstance(frame, np.ndarray):
                frames.append(annotate_with_time(frame, sim_time))
                dt = sim_time - last_kept_time
                durations.append(quantize_duration(dt * slowmo))
                last_kept_time = sim_time
            else:
                print(f"Warning: got non-array frame: {type(frame)}")

        t += 1
        if stop_on_first_episode and ts.last():
            if include_terminal_frame:
                frame = grab_frame()
                if isinstance(frame, np.ndarray):
                    sim_time = hooked.physics.data.time
                    frames.append(annotate_with_time(frame, sim_time))
                    durations.append(quantize_duration((sim_time - last_kept_time) * slowmo))
            print(f"step {t} - episode ended after {t} steps, time={sim_time:.2f}s")
            break

    print(f"Captured {len(frames)} frames over {hooked.physics.data.time:.3f}s sim time.")
    if len(frames) <= 1:
        raise RuntimeError("Not enough frames captured; try longer 'steps' or larger 'viewer_min_delay_ms'.")

    # ensure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    imageio.mimsave(out_path, frames, duration=durations, loop=0)
    print(f"[OK] Saved GIF: {out_path}  frames={len(frames)}  "
          f"sim={hooked.physics.data.time:.3f}s  sum_dur={sum(durations):.3f}s  "
          f"slowmo={slowmo}  viewer_min_delay_ms={viewer_min_delay_ms}")
