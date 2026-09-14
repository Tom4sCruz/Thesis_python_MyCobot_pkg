# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`armik` — a from-scratch Cartesian IK control layer for the myCobot 280 (Jetson Nano), built on `pymycobot`, for a university thesis on human-likeness in robot-arm reaching motions. `pymycobot` is used only as a serial codec (`send_angles()`/`send_angle()`); this package reimplements FK/IK/planning in Python because the firmware's own `send_coords()` has no partial constraints, fails silently on unreachable targets, and can't be inspected or debugged.

The whole repo is one Python package, `thesis_armik_pkg/` — there is no other subproject.

## Commands

Install (from `thesis_armik_pkg/`):
```bash
pip install numpy pymycobot
pip install -e .
```

First-time hardware bring-up, in this order (not optional — step 2 validates the DH table against the physical arm):
```bash
python3 scripts/test_arm.py --mock --yes             # 1. maths + flow, no hardware
python3 scripts/verify_fk.py --port /dev/ttyTHS1     # 2. does the DH table match THIS arm?
python3 scripts/test_arm.py --port /dev/ttyTHS1      # 3. does it actually move?
```

There is no pytest/unittest suite and no CI. Validation is done by standalone, self-contained scripts in `scripts/` (`test_arm.py`, `test_lock.py`, `test_path.py`, `test_gripper_tcp.py`, `verify_fk.py`, ...), each printing its own PASS/FAIL stage banners; most support `--mock` (no hardware) and `--yes` (skip the confirmation prompt). There's no "run a single test" — each script *is* one suite; run the whole script.

Movement-profile scripts (the current focus of thesis work) live in `scripts/Profiles/` and run the same way:
```bash
python3 scripts/Profiles/<name>.py --mock --yes            # simulate, no hardware
python3 scripts/Profiles/<name>.py --port /dev/ttyTHS1     # real hardware
python3 scripts/Profiles/<name>.py --mock --yes --rviz     # simulate + stream to RViz2 (needs Study-docker, see below)
```

## Architecture

### `armik/` package — read in this order (bottom-up dependency order, per its own README)

1. `config.py` — DH table, joint limits, and every tuning constant (single-joint timing, jerk dials, IK weights). **Read this first** — almost all package behavior is a constant here, not code.
2. `kinematics.py` — FK, geometric/task Jacobian, Euler↔matrix, `check_joint_limits`/`check_workspace_bounds`.
3. `ik.py` — weighted damped-least-squares IK with partial (per-axis) constraints.
4. `connection.py` — `ArmConnection`: thread-safe wrapper around pymycobot (or the mock), converts float deg/s to the firmware's integer 0–100 speed field.
5. `mock.py` — `MockMyCobot`, an offline stand-in with instantaneous/perfect tracking, used whenever `--mock` is passed.
6. `blending.py` — Hermite multi-waypoint blending, used by `Arm.send_path`.
7. `jerk.py` — `JerkInjector`, the deliberate-jitter engine (tremor/twitch/stutter) — the *opposite* of the package's internal min-jerk smoothing, used to make motion visibly shaky for comparison. All-zero dials = byte-identical smooth motion.
8. `arm.py` — the public API, class `Arm`.

### `Arm` (`armik/arm.py`)

Talks to hardware only through `ArmConnection`, guarded by a single lock. Two motion regimes, toggled with `set_single_joint(bool)`:
- **Blended/Cartesian** (`_execute`): streams a planned path at `config.CONTROL_RATE_HZ` on an absolute-deadline schedule.
- **Single-joint** (`_execute_single_joint` / `_drive_joint`): moves one joint at a time, blocking until arrival. `_drive_joint` re-sends a stalled command (stall detected via `SINGLE_JOINT_STALL_S`/`SINGLE_JOINT_PROGRESS_DEG`, up to `SINGLE_JOINT_RESEND_MAX` retries) to work around a firmware quirk where a motion packet is occasionally dropped.

Every plan is validated end-to-end (`plan_coords`/`plan_path`: IK-solve → check → stream) before anything is sent to hardware — planning and execution are deliberately separate steps.

Jerk dials live directly on the `Arm` instance (`arm.jerk`, `arm.random_twitch`, `arm.twitch_intensity`, `arm.jerk_seed`), backed by one persistent `JerkInjector`. In single-joint mode there's no control-rate stream to carry a tremor, so jerk instead shows up as a "stutter": `config.STUTTER_TYPE` selects a lateral WOBBLE (jittered sub-commands before a clean settle) vs. a STOP-and-go hesitation (the joint halts at random points mid-travel via `conn.stop()`, then resumes) — both implemented inside `_drive_joint`'s poll loop together with `_jerk_stutter()`.

### Units (easy to get wrong)

Internally everything is **mm and degrees**. The public `Arm` API (`send_coords`/`get_coords`) uses **cm** for position, degrees for orientation/joints, cm/s for speed — the conversion happens only in `_cm_to_mm`/`_mm_to_cm` in `arm.py`. **Naming hazard:** `arm.send_coords(...)` (cm, partial constraints) vs. `arm.conn.raw.send_coords(...)` (pymycobot's own, mm, full pose only) — confusing the two is a silent 10x error, not a crash.

`config.JOINT_1_HEIGHT_CM` is documentation-only — do **not** add it to z; the DH table's `d1` already accounts for the base height (see the comment above it in `config.py`).

### Movement-profile scripts (`scripts/Profiles/`)

Standalone and deliberately copy-pasted rather than sharing a base class — each script's docstring says which sibling it "mirrors." Common shape: CONSTANTS → geometry/state-check helpers → MOTION functions (drive the `Arm` directly) → PREFLIGHT (dry-run every waypoint with `plan_coords`/`plan_path` before any real motion) → MAIN (argparse, build segments, instantiate `Arm`, optionally wire up `RvizBridge`, run the cycle loop, home, `finally: arm.close()`). All import `armik` via a `sys.path.insert` hack (they sit two levels below the package root).

Current family, by choreography:
- `LowH-HighR.py` — single-joint SWING(J1) / DESCEND / LIFT(J2..J4).
- `LowH-HighR_v2.py` — coordinated grab: J1+J6 co-rotated swing, multi-joint lift/descent.
- `LowH-HighR_v3.py` — like v1 but lift is J2-only.
- `HighH-LowR.py` — randomized parabolic arcs, jittered apex/position, random pick order, scripted "recoil" nudge.
- `HighH-LowR_v2.py` — same arcs, J6 hard-frozen at HOME for the whole run.
- `HighH-HighR.py` — deterministic clone of `HighH-LowR.py`: fixed apex (`APEX_Z_CM`), constant tip speed, no randomness — same every run.

Shared dials worth knowing: `NUDGE_CYCLE` (which "reach" cycle triggers a scripted interruption; `-1` disables) appears in every profile script; `STUTTER_TYPE` lives centrally in `armik/config.py` rather than per-script.

### RViz mock visualization (`--rviz`)

Mock-only — profile scripts force it off unless `--mock` is also passed. `scripts/Profiles/_rviz_bridge.py` (`RvizBridge`) is the only place `rclpy` is imported anywhere in the repo, guarded so `armik` itself stays ROS-free. It does not launch RViz itself — it only publishes `/joint_states` (polled from the mock `Arm`'s live state — deliberately not `joint_state_publisher`, to avoid a duplicate publisher) and `visualization_msgs/Marker` (planned path, cube markers, live tip). RViz2 + `robot_state_publisher` are started separately via `thesis_armik_pkg/ros/launch/mock_display.launch.py`, using the URDF/RViz config under `thesis_armik_pkg/ros/`.

This needs `rclpy`, which only exists inside the sibling **Study-docker** `hri_thesis` container (ROS2 Humble; `docker-compose.yml`/`enter-docker.sh` live in `../Study-docker/`, one directory above this repo, which is bind-mounted read-write into the container at `/thesis`). Run sequence inside the container:
```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws && colcon build --symlink-install --packages-select mycobot_description && source install/setup.bash
ros2 launch /thesis/thesis_armik_pkg/ros/launch/mock_display.launch.py
# in a second shell:
cd /thesis/thesis_armik_pkg
python3 scripts/Profiles/HighH-LowR.py --mock --yes --rviz
```
