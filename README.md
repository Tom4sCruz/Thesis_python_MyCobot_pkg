# armik

Cartesian control for the myCobot 280 (Jetson Nano), built directly on
**pymycobot**. Motion is issued only through `send_angles()` / `send_angle()`;
the firmware's `send_coords()` / `send_coord()` are never used.

## Why not just use the firmware's send_coords()

pymycobot does no kinematics. It is a serial protocol codec: it validates
arguments, converts units to the wire encoding, frames packets
(`0xFE 0xFE ... 0xFA`), writes them, and decodes replies. Everything that looks
like kinematics — `get_coords()`, `send_coords()`, `solve_inv_kinematics()` —
is forwarded to firmware you cannot inspect, constrain, or debug.

That produces three problems this package fixes:

| Firmware `send_coords()` | `armik` |
|---|---|
| Demands a full 6-DOF pose every time | Constrain any subset; the rest stay put |
| Fails silently on unreachable targets | Returns 0 with the reason in `last_error` |
| No way to hold a joint or bound the path | Whole path validated before anything is sent |

## Layout

```
armik/
    config.py       DH table, joint limits, tuning constants   <- verify these first
    kinematics.py   FK, geometric Jacobian, Euler, task Jacobian
    ik.py           weighted damped-least-squares IK with partial constraints
    connection.py   pymycobot wrapper: locking, retries, float->int speed
    mock.py         offline stand-in for MyCobot280
    arm.py          the public API: get_coords(), send_coords()
scripts/
    verify_fk.py    validate the DH table against firmware get_coords()
    test_arm.py     staged validation suite
    example.py      template for your own scripts
```

Read them in that order — it is the dependency order, bottom-up.

## Install

```bash
pip install numpy pymycobot
cd armik_pkg
pip install -e .
```

## Order of operations, first time

```bash
python3 scripts/test_arm.py --mock --yes             # 1. maths + flow, no hardware
python3 scripts/verify_fk.py --port /dev/ttyTHS1     # 2. does DH match YOUR arm?
python3 scripts/test_arm.py --port /dev/ttyTHS1      # 3. does it actually move?
```

Step 2 is not optional. Every Cartesian number is only as correct as the DH
table, and the table shipped here is a starting point, not ground truth.

## Usage

```python
from armik import Arm

with Arm(port="/dev/ttyTHS1") as arm:
    if not arm.conn.is_power_on():
        arm.conn.power_on()

    arm.get_coords()                              # [x,y,z (cm), rx,ry,rz (deg)]

    arm.send_coords(x=20, y=-6, z=15, speed=4.0)  # position; orientation free
    arm.send_coords(z=12)                          # z only; everything else held
    arm.send_coords(x=18, rz=45, speed=2.0)        # mixed

    if arm.send_coords(x=25) == 0:
        print(arm.last_error)                      # why it refused
```

Every coordinate is optional. Pass what you want held; omit the rest.

### Partial constraints and minimal drift

An omitted coordinate is **not** "don't care". It is soft-anchored to the value
it had when the motion started, at `config.FREE_ANCHOR_WEIGHT`. It prefers to
stay put and only yields when the constrained coordinates genuinely need it to.

Measured on a 3 cm z-only move from home:

| `FREE_ANCHOR_WEIGHT` | x drift | rx drift |
|---|---|---|
| 0.08 | 3.2 mm | 11.4° |
| **2.0** (default) | **0.03 mm** | **0.22°** |
| 8.0 | 0.001 mm | 0.01° |

Raise it to pin the free coordinates harder; lower it if a reachable target
starts being reported unreachable because the anchors are fighting the
constraints. `plan.free_drift` reports what actually moved on every call.

### Plan without moving

```python
plan = arm.plan_coords(x=25.0, speed=4.0)
if plan.ok:
    print(plan.path_length_cm, plan.peak_joint_dps, plan.free_drift)
else:
    print(plan.error)
```

## Units

| quantity | unit |
|---|---|
| `send_coords` / `get_coords` position | **centimetres** |
| orientation (`rx`, `ry`, `rz`) | degrees |
| `speed` in `send_coords` | **cm/s** of tip travel |
| joint angles | degrees |
| everything inside the package | mm and degrees |

The cm conversion happens only in `_cm_to_mm` / `_mm_to_cm` in `arm.py`, so
there is one place to look when a number is off by 10x.

**Naming hazard:** `arm.send_coords(...)` takes cm and partial constraints;
`arm.conn.raw.send_coords(...)` is pymycobot's, takes mm and a full pose.
Confusing them is a silent 10x error, not a crash.

## The base height

`config.JOINT_1_HEIGHT_CM` exists for documentation. **Do not add it to z.**

`DH_TABLE` row 1 already has `d1 = 131.56 mm = 13.156 cm`, which puts frame 0 at
the bottom of the base and frame 1 at the joint-1 axis. The z from
`get_coords()` is therefore *already* measured from the table. Adding the
constant would put every target ~13 cm too high.

A tape measure to the top of the base housing reads ~15 cm; the kinematically
meaningful figure is the distance to the joint-1 axis, 13.156 cm. Resolve that
~1.8 cm gap with `verify_fk.py` before trusting Cartesian targets. If you want z
measured from the joint-1 axis instead of the table, set
`config.Z_RELATIVE_TO_JOINT1 = True`.

## Known limits

- **Open loop.** The firmware runs the servo loop; this feeds it setpoints and
  logs what was commanded. There is no outer position correction.
- **Gimbal lock.** Orientation goes through Euler angles — necessary to
  constrain `rx`/`ry`/`rz` independently — which degenerates near `ry = ±90°`.
  A warning is logged within `config.GIMBAL_WARN_DEG` of it.
- **Straight line applies to constrained axes.** Constrain only `x` and the tip
  travels linearly in x while y and z make minimal adjustments.
- **`DEG_PER_S_AT_SPEED_100` is a guess** until you measure it. It only affects
  the firmware speed field, not path geometry, but it makes reported velocities
  fictional until calibrated.
- **Bare flange assumed.** Mount a gripper and every Cartesian target is off by
  the tool length until you add the offset to the DH table.
