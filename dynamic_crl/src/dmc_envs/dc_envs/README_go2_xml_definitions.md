### Where do action and observation spaces come from?

In MuJoCo / dm_control, the **robot XML model** (e.g. `go2.xml`) defines the physical body,
joints, and actuators. From this file:

- **Action space**  
  Each `<motor>` or `<general>` actuator in the XML contributes one dimension.  
  The `ctrlrange="min max"` attribute defines the valid range.  
  `dm_control` builds an `action_spec` from this, which we convert into a
  `gym.spaces.Box` for compatibility.

- **Observation space**  
  By default, `dm_control` exposes a dictionary of sensor readings
  (joint positions, velocities, body sensors, etc.) as defined in the XML model.  
  In our wrapper, these are flattened into a single continuous vector (`Box`).

So the XML fixes the **mechanics and control interface** (actions + raw observations).  
On top of this, we are free to define the **task**:  
- Reward function  
- Episode termination / truncation conditions  
- Context features (gravity, friction, payload, latency, etc.)  
- Any scheduling or perturbations during training

In short: the XML gives you the robot’s body and its low-level interface,  
and you decide *what task or environment logic to wrap around it*.


# Go2 Environment Setup Plan

When building tasks on top of the `CARLDmcGo2Env`, we should cover the following points.  
The XML defines the **actuators (actions)** and **sensors (observations)**, but task logic and interfaces are up to us.  

---

## 1. Lock control rate & decimation
- **What**: Decide physics `dt` (e.g., 0.005 s) and apply `decimation` (e.g., 4 steps per action) so the policy acts at 50 Hz.  
- **Why**: The real robot has a fixed control frequency; mismatch causes instability at deployment.  
- **Difficulty**: Low — just repeat the action for `decimation` steps.  
- **Importance**: High — must match reality.

---

## 2. Pick a control interface
- **What**: Either (a) send torques directly (from XML actuators), or (b) wrap actions with a **PD adapter** that maps normalized actions → position targets → torques.  
- **Why**: Real Go2 often runs in position-PD mode, not raw torque. We must align sim with real hardware.  
- **Difficulty**: Medium — PD shim is ~20 lines of code, but tuning gains is non-trivial.  
- **Importance**: High — defines how the policy talks to the robot.

---

## 3. Normalize actions & observations
- **What**:  
  - Actions: policy outputs in `[-1,1]`; scale/clamp to actuator ranges.  
  - Observations: flatten via `flat_observation=True` (already done) and normalize (z-score or fixed scale).  
- **Why**: Learning stability. Ensures values are in comparable ranges across sim and real.  
- **Difficulty**: Low — one preprocessing layer.  
- **Importance**: High — critical for training and sim→real transfer.

---

## 4. Start with a simple, testable task
- **What**: Implement **velocity tracking**: robot should follow a desired forward speed.  
  - Reward = tracking accuracy − energy usage + uprightness.  
  - Terminate if base height/tilt exceed limits.  
- **Why**: Standard baseline in legged locomotion; tests whether env is wired up correctly.  
- **Difficulty**: Medium — requires reward shaping and logging.  
- **Importance**: High — foundation for all further tasks.

---

## 5. Seed & determinism hooks
- **What**: Support `reset(seed=...)` fully; log seeds and contexts.  
- **Why**: Reproducibility for experiments and debugging.  
- **Difficulty**: Low.  
- **Importance**: Medium (but essential for research workflow).

---

## 6. Latency & payload toggles
- **What**: Enable context features like `latency_steps` and `payload_mass` after baseline works.  
- **Why**: Domain randomization and robustness — prepares for real-world disturbances.  
- **Difficulty**: Low — you scaffolded this already.  
- **Importance**: Medium — not needed for first baseline, but valuable later.

---

## 7. Logging & sanity checks
- **What**: Log per-step metrics: action magnitudes, torque saturation, reward terms, base height, velocity error.  
- **Why**: Debugging, reward shaping, and making sure the policy is “learning the right thing.”  
- **Difficulty**: Low — just extend `info` dict.  
- **Importance**: High — saves time in debugging.

---

## 8. Minimal viewer path
- **What**: Stick to `render(mode="rgb_array")` and matplotlib for inspection; skip the interactive viewer for now.  
- **Why**: Keeps code lightweight; avoids dm_control viewer bugs during setup.  
- **Difficulty**: Low.  
- **Importance**: Low (nice-to-have visualization, not essential for training).

---

# Summary of Priorities

| Task                        | Difficulty | Importance |
|-----------------------------|------------|------------|
| Control rate & decimation   | Low        | High       |
| Control interface (torque/PD) | Medium   | High       |
| Normalize actions/obs       | Low        | High       |
| Velocity tracking task      | Medium     | High       |
| Seeds & determinism         | Low        | Medium     |
| Latency & payload toggles   | Low        | Medium     |
| Logging & sanity checks     | Low        | High       |
| Minimal viewer              | Low        | Low        |

**Bottom line:**  
The XML gives us the robot’s body and low-level interface. Our job is to add a consistent **control interface, scaling, task definition, and logging**. Start with a simple velocity-tracking reward and PD control adapter, then layer on complexity (latency, payloads, terrains).
