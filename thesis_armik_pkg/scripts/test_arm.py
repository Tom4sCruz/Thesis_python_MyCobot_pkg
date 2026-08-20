#!/usr/bin/env python3
"""
STEP 2 -- does it work, and does it fail safely?

    python3 scripts/test_arm.py --mock --yes        # no hardware needed
    python3 scripts/test_arm.py --port /dev/ttyTHS1

Stages:
  1  connection and state
  2  kinematics self-check (no motion)
  3  get_coords() and the base-height sanity check
  4  full 6-DOF constraint
  5  position only, rotations free
  6  single axis -- the minimal-drift test
  7  straight-line verification
  8  failure modes -- the arm must NOT move

CLEAR THE WORKSPACE BEFORE RUNNING AGAINST REAL HARDWARE.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging

import numpy as np

from armik import Arm, config, forward_kinematics, ik, kinematics, pose_coords
from armik.kinematics import wrap180

HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]


def banner(n, title):
    print(f"\n{'=' * 68}\n[{n}] {title}\n{'=' * 68}")


def check(cond, msg):
    print(f"    {'PASS' if cond else 'FAIL'}  {msg}")
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--csv", default=None, help="write the last trajectory here")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.mock and not args.yes:
        print("This will move the robot arm. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    failures = []
    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)

    def home():
        arm.move_joints(HOME, duration=2.5)

    # ------------------------------------------------------------------
    banner(1, "Connection and state")
    if not arm.conn.is_power_on():
        print("    powering on...")
        arm.conn.power_on()
    if not check(arm.conn.is_power_on(), "arm reports power on"):
        arm.close()
        return 1
    q = arm.get_angles()
    check(len(q) == config.DOF, f"read 6 joint angles: {[round(v,1) for v in q]}")

    # ------------------------------------------------------------------
    banner(2, "Kinematics self-check (nothing moves)")
    qt = np.array([15.0, -25.0, -60.0, -20.0, 10.0, 5.0])

    J = kinematics.geometric_jacobian(qt)
    Jn = np.zeros((6, 6))
    T = forward_kinematics(qt)
    h = 1e-6
    for i in range(6):
        qp = qt.copy()
        qp[i] += np.rad2deg(h)
        Tp = forward_kinematics(qp)
        Jn[:3, i] = (Tp[:3, 3] - T[:3, 3]) / h
        dR = Tp[:3, :3] @ T[:3, :3].T
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if ang > 1e-12:
            v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            Jn[3:, i] = v * ang / (2 * np.sin(ang)) / h
    check(np.abs(J - Jn).max() < 1e-3,
          f"geometric Jacobian matches numerical ({np.abs(J-Jn).max():.2e})")

    c = pose_coords(qt)
    R2 = kinematics.rpy_to_matrix(c[3], c[4], c[5])
    check(np.abs(R2 - T[:3, :3]).max() < 1e-9, "Euler round-trip is exact")

    des = c.copy()
    des[1] += 30
    des[2] -= 20
    sol = ik.try_solve(des, np.ones(6, bool), qt)
    if check(sol is not None, "IK solves a full 6-DOF target"):
        got = pose_coords(sol)
        check(np.abs(got[:3] - des[:3]).max() < config.POS_TOL_MM * 2,
              f"IK position residual {np.abs(got[:3]-des[:3]).max():.3f} mm")
    else:
        failures.append("ik")

    # ------------------------------------------------------------------
    banner(3, "get_coords() and the base-height check")
    home()
    s = arm.get_coords()
    print(f"    coords (cm, deg): {s}")
    check(len(s) == 6, "returns 6 values")
    z_dh = config.DH_TABLE[0][1] / 10.0
    print(f"    DH d1 = {z_dh:.3f} cm, JOINT_1_HEIGHT_CM = {config.JOINT_1_HEIGHT_CM:.3f} cm")
    print("    NOTE: FK already includes the base height. z is measured from the")
    print("          table, so JOINT_1_HEIGHT_CM must NOT be added to it again.")
    if not args.mock:
        fw = arm.conn.get_coords_firmware()
        d = abs(fw[2] / 10.0 - s[2])
        check(d < 1.0, f"our z agrees with firmware z within 1 cm ({d:.2f} cm)")
        if d >= 1.0:
            print("    -> run scripts/verify_fk.py")

    # ------------------------------------------------------------------
    banner(4, "Full 6-DOF constraint")
    home()
    s = arm.get_coords()
    tgt = [s[0], s[1] + 2.0, s[2] - 1.5, s[3], s[4], s[5]]
    r = arm.send_coords(*tgt, speed=4.0)
    if not check(r == 1, "send_coords with all six constrained"):
        failures.append("6dof")
        print(f"    {arm.last_error}")
    else:
        g = arm.get_coords()
        pe = max(abs(g[i] - tgt[i]) for i in range(3))
        re = max(abs(float(wrap180(g[i] - tgt[i]))) for i in range(3, 6))
        check(pe < 0.2, f"position within 2 mm ({pe*10:.2f} mm)")
        check(re < 1.5, f"orientation within 1.5 deg ({re:.2f} deg)")

    # ------------------------------------------------------------------
    banner(5, "Position only -- rotations free")
    home()
    s = arm.get_coords()
    r = arm.send_coords(x=s[0], y=s[1] + 3.0, z=s[2] - 2.0, speed=4.0)
    if not check(r == 1, "send_coords(x, y, z)"):
        failures.append("xyz")
        print(f"    {arm.last_error}")
    else:
        print(f"    free-coord drift: {arm.last_plan.free_drift}")
        g = arm.get_coords()
        pe = max(abs(g[i] - t) for i, t in
                 zip(range(3), [s[0], s[1] + 3.0, s[2] - 2.0]))
        check(pe < 0.2, f"position within 2 mm ({pe*10:.2f} mm)")

    # ------------------------------------------------------------------
    banner(6, "Single axis -- the minimal-drift test")
    home()
    s = arm.get_coords()
    r = arm.send_coords(z=s[2] - 3.0, speed=4.0)
    if not check(r == 1, "send_coords(z=...) alone"):
        failures.append("z-only")
        print(f"    {arm.last_error}")
    else:
        drift = arm.last_plan.free_drift
        print(f"    free-coord drift: {drift}")
        wp = max(abs(drift.get(k, 0.0)) for k in ("x", "y"))
        wr = max(abs(drift.get(k, 0.0)) for k in ("rx", "ry", "rz"))
        check(wp < 0.2, f"x/y held within 2 mm ({wp*10:.2f} mm)")
        check(wr < 2.0, f"rotations held within 2 deg ({wr:.2f} deg)")

    # ------------------------------------------------------------------
    banner(7, "Straight-line verification")
    home()
    s = arm.get_coords()
    r = arm.send_coords(x=s[0] - 1.5, y=s[1] + 4.0, z=s[2] - 3.0, speed=4.0)
    if not check(r == 1, "diagonal move executed"):
        failures.append("line")
        print(f"    {arm.last_error}")
    else:
        tips = np.array([forward_kinematics(q)[:3, 3]
                         for q in arm.last_plan.q_waypoints])
        p0, p1 = tips[0], tips[-1]
        vec = p1 - p0
        L = float(np.linalg.norm(vec))
        u = vec / L
        rel = tips - p0
        perp = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        print(f"    {L/10:.2f} cm over {len(tips)} waypoints, "
              f"peak joint speed {arm.last_plan.peak_joint_dps:.1f} deg/s")
        check(perp.max() < 2.0,
              f"max deviation from the line {perp.max():.3f} mm "
              f"({perp.max()/L*100:.3f}% of path)")
        ex = arm.last_execution
        print(f"    {ex.setpoints} setpoints, {ex.late_deadlines} late, "
              f"took {ex.duration_s:.2f}s (planned {arm.last_plan.duration_s:.2f}s)")
        if args.csv:
            ex.to_csv(args.csv)
            print(f"    trajectory written to {args.csv}")

    # ------------------------------------------------------------------
    banner(8, "Failure modes -- the arm must NOT move")
    home()
    before = arm.get_angles()
    cases = [
        ("far out of reach", dict(x=60.0)),
        ("no constraints given", dict()),
        ("negative speed", dict(z=15.0, speed=-1.0)),
        ("speed too high for hardware", dict(x=22.0, speed=200.0)),
    ]
    for name, kw in cases:
        rr = arm.send_coords(**kw)
        check(rr == 0, f"refused: {name}")
        print(f"        reason: {arm.last_error}")
    after = arm.get_angles()
    moved = max(abs(a - b) for a, b in zip(after, before))
    if not check(moved < 0.5, f"arm did not move during any refusal ({moved:.4f} deg)"):
        failures.append("silent-failure")

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("FAILED: " + ", ".join(failures) if failures else "All stages passed.")
    home()
    arm.close()
    return 1 if failures else 0


if __name__ == "__main__":
    _sys.exit(main())
