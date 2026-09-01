#!/usr/bin/env python3
"""
Validation for the gripper command and the tool-tip (TCP) offset.

    python3 scripts/test_gripper_tcp.py --mock --yes        # no hardware needed
    python3 scripts/test_gripper_tcp.py --port /dev/ttyTHS1

Stages:
  1  connection and state
  2  send_gripper() -- value mapping, speed passthrough, refusals
  3  tool_transform() math -- offset + roll, vs the bare flange
  4  zero-offset regression -- FK is byte-identical to the flange when unset
  5  get_coords() is measured to the gripper tip, not the flange
  6  send_coords() round-trip lands the TIP where asked
  7  workspace guard still refuses an out-of-reach tip target
  8  hardware only -- the gripper actuates, and a level gripper reads rz ~= 0

CLEAR THE WORKSPACE BEFORE RUNNING AGAINST REAL HARDWARE.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import logging

import numpy as np

from armik import Arm, config, forward_kinematics, pose_coords
from armik.kinematics import (
    flange_pose_coords,
    frame_chain,
    matrix_to_rpy,
    rpy_to_matrix,
    tool_transform,
)

HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
TEST_Q = [
    np.array([0.0, 0.0, -90.0, 0.0, 0.0, 0.0]),
    np.array([15.0, -25.0, -60.0, -20.0, 10.0, 5.0]),
    np.array([-30.0, 10.0, -60.0, -30.0, -20.0, 25.0]),
    np.array([45.0, -40.0, -50.0, 0.0, 40.0, -30.0]),
]


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
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.mock and not args.yes:
        print("This will move the robot arm and actuate the gripper. Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    failures = []
    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)

    def home():
        arm.move_joints(HOME, duration=1.5)

    # ------------------------------------------------------------------
    banner(1, "Connection and state")
    if not arm.conn.is_power_on():
        print("    powering on...")
        arm.conn.power_on()
    if not check(arm.conn.is_power_on(), "arm reports power on"):
        arm.close()
        return 1
    print(f"    TOOL_OFFSET_MM = {config.TOOL_OFFSET_MM}, "
          f"TOOL_RPY_DEG = {config.TOOL_RPY_DEG}, "
          f"MAX_GRIPPER_DEG = {config.MAX_GRIPPER_DEG:g}")

    # ------------------------------------------------------------------
    banner(2, "send_gripper() -- mapping, speed, refusals")
    md = config.MAX_GRIPPER_DEG
    cases_ok = [
        (md, 100),          # fully open
        (0.0, 0),           # closed
        (md / 2.0, 50),     # halfway
    ]
    for deg, want_val in cases_ok:
        r = arm.send_gripper(deg)
        ok = r == 1
        if args.mock and ok:
            got = arm.conn.raw.gripper_commands[-1][1]
            ok = got == want_val
            check(ok, f"send_gripper({deg:g}) -> value {got} (want {want_val})")
        else:
            check(ok, f"send_gripper({deg:g}) returned {r}")
        if not ok:
            failures.append("gripper-map")

    if args.mock:
        arm.send_gripper(md / 2.0, speed=30)
        got_speed = arm.conn.raw.gripper_commands[-1][2]
        if not check(got_speed == 30, f"speed passthrough: recorded {got_speed} (want 30)"):
            failures.append("gripper-speed")

    for name, kw in [
        ("deg above MAX_GRIPPER_DEG", dict(deg=md + 10.0)),
        ("negative deg", dict(deg=-5.0)),
        ("speed out of 1-100", dict(deg=md / 2.0, speed=999)),
    ]:
        r = arm.send_gripper(**kw)
        if not check(r == 0, f"refused: {name}"):
            failures.append("gripper-refuse")
        else:
            print(f"        reason: {arm.last_error}")

    # ------------------------------------------------------------------
    banner(3, "tool_transform() math (offset + roll vs bare flange)")
    T_tool = tool_transform()
    off = np.array(config.TOOL_OFFSET_MM, dtype=float)
    R_off = rpy_to_matrix(*config.TOOL_RPY_DEG)
    check(np.allclose(T_tool[:3, 3], off, atol=1e-9), f"translation block == {tuple(off)}")
    check(np.allclose(T_tool[:3, :3], R_off, atol=1e-9), "rotation block == rpy_to_matrix(TOOL_RPY_DEG)")

    worst_pos = worst_rot = worst_flange = 0.0
    for q in TEST_Q:
        Tf = frame_chain(q)[-1]
        expect_pos = Tf[:3, 3] + Tf[:3, :3] @ off
        expect_rot = matrix_to_rpy(Tf[:3, :3] @ R_off)
        pc = pose_coords(q)
        worst_pos = max(worst_pos, float(np.max(np.abs(pc[:3] - expect_pos))))
        worst_rot = max(worst_rot, float(np.max(np.abs(pc[3:] - expect_rot))))
        worst_flange = max(worst_flange, float(np.max(np.abs(flange_pose_coords(q)[:3] - Tf[:3, 3]))))
    check(worst_pos < 1e-6, f"pose_coords tip position matches flange + R@offset ({worst_pos:.2e} mm)")
    check(worst_rot < 1e-6, f"pose_coords tip orientation matches flange @ roll ({worst_rot:.2e} deg)")
    check(worst_flange < 1e-9, f"flange_pose_coords still == frame_chain[-1] ({worst_flange:.2e} mm)")

    # geometric Jacobian must still match a numerical difference of the (tip) FK
    qj = TEST_Q[1]
    from armik import geometric_jacobian
    Jg = geometric_jacobian(qj)
    Jn = np.zeros((6, 6))
    T = forward_kinematics(qj)
    h = 1e-6
    for i in range(6):
        qp = qj.copy(); qp[i] += np.rad2deg(h)
        Tp = forward_kinematics(qp)
        Jn[:3, i] = (Tp[:3, 3] - T[:3, 3]) / h
        dR = Tp[:3, :3] @ T[:3, :3].T
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if ang > 1e-12:
            v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            Jn[3:, i] = v * ang / (2 * np.sin(ang)) / h
    if not check(np.abs(Jg - Jn).max() < 1e-3,
                 f"geometric Jacobian matches numerical tip FK ({np.abs(Jg-Jn).max():.2e})"):
        failures.append("jacobian")

    # ------------------------------------------------------------------
    banner(4, "Zero-offset regression -- FK identical to the flange when unset")
    saved_off, saved_rpy = config.TOOL_OFFSET_MM, config.TOOL_RPY_DEG
    try:
        config.TOOL_OFFSET_MM = (0.0, 0.0, 0.0)
        config.TOOL_RPY_DEG = (0.0, 0.0, 0.0)
        exact = all(np.array_equal(forward_kinematics(q), frame_chain(q)[-1]) for q in TEST_Q)
        same_pc = all(np.array_equal(pose_coords(q), flange_pose_coords(q)) for q in TEST_Q)
        check(np.array_equal(tool_transform(), np.eye(4)), "tool_transform() == eye(4)")
        if not check(exact, "forward_kinematics() byte-identical to frame_chain[-1]"):
            failures.append("regression")
        if not check(same_pc, "pose_coords() byte-identical to flange_pose_coords()"):
            failures.append("regression")
    finally:
        config.TOOL_OFFSET_MM, config.TOOL_RPY_DEG = saved_off, saved_rpy
    check(not np.array_equal(tool_transform(), np.eye(4)) or saved_off == (0.0, 0.0, 0.0),
          "tool_transform() restored to the configured value")

    # ------------------------------------------------------------------
    banner(5, "get_coords() is measured to the gripper tip")
    home()
    q_now = np.array(arm.get_angles(), dtype=float)
    g = np.array(arm.get_coords(), dtype=float)              # cm / deg, tip
    tip = pose_coords(q_now)                                  # mm / deg, tip
    flange = flange_pose_coords(q_now)                        # mm / deg, flange
    exp = tip.copy()
    exp[:3] /= 10.0
    if config.Z_RELATIVE_TO_JOINT1:
        exp[2] -= config.JOINT_1_HEIGHT_CM
    check(np.max(np.abs(g[:3] - exp[:3])) < 1e-3, "get_coords position == tip position")
    check(np.max(np.abs(g[3:] - exp[3:])) < 1e-2, "get_coords orientation == tip orientation")
    tip_minus_flange_cm = (tip[:3] - flange[:3]) / 10.0
    expect_shift = frame_chain(q_now)[-1][:3, :3] @ (np.array(config.TOOL_OFFSET_MM) / 10.0)
    check(np.max(np.abs(tip_minus_flange_cm - expect_shift)) < 1e-4,
          f"tip is offset from the flange by R_flange @ {tuple(np.array(config.TOOL_OFFSET_MM)/10.0)} cm")

    # ------------------------------------------------------------------
    banner(6, "send_coords() round-trip lands the TIP where asked")
    home()
    s = arm.get_coords()
    target = [s[0], s[1] + 2.0, s[2] - 1.5]
    r = arm.send_coords(x=target[0], y=target[1], z=target[2], speed=4.0)
    if not check(r == 1, "send_coords(x, y, z) to a nearby tip target"):
        failures.append("roundtrip")
        print(f"    {arm.last_error}")
    else:
        g = arm.get_coords()
        pe = max(abs(g[i] - target[i]) for i in range(3))
        check(pe < config.POS_TOL_MM * 2 / 10.0, f"tip within {config.POS_TOL_MM*2:.1f} mm ({pe*10:.2f} mm)")
        tip_end = forward_kinematics(arm.last_plan.q_waypoints[-1])[:3, 3]
        want_mm = np.array(target) * 10.0
        if config.Z_RELATIVE_TO_JOINT1:
            want_mm[2] += config.JOINT_1_HEIGHT_CM * 10.0
        check(np.max(np.abs(tip_end - want_mm)) < config.POS_TOL_MM,
              f"planned path ends at the tip target ({np.max(np.abs(tip_end-want_mm)):.3f} mm)")

    # ------------------------------------------------------------------
    banner(7, "Workspace guard still refuses an out-of-reach tip target")
    home()
    before = arm.get_angles()
    r = arm.send_coords(x=70.0, speed=4.0)
    check(r == 0, f"refused x=70 cm ({arm.last_error})")
    after = arm.get_angles()
    if not check(max(abs(a - b) for a, b in zip(after, before)) < 0.5, "arm did not move on the refusal"):
        failures.append("guard")

    # ------------------------------------------------------------------
    banner(8, "Hardware -- gripper actuates; level gripper reads rz ~= 0")
    if args.mock:
        print("    (skipped in --mock)")
    else:
        check(arm.send_gripper(config.MAX_GRIPPER_DEG) == 1, "send_gripper(MAX) -- gripper should open")
        input("    watch the gripper, then press enter to close...")
        check(arm.send_gripper(0.0) == 1, "send_gripper(0) -- gripper should close")
        print("    Jog / move the arm until the gripper is visibly parallel to the ground,")
        print("    then read get_coords():")
        input("    press enter when the gripper is level...")
        lvl = arm.get_coords()
        print(f"    get_coords() = {lvl}")
        if not check(abs(float(lvl[5])) < 8.0, f"rz ~= 0 when the gripper is level (rz={lvl[5]:.2f})"):
            print("    -> adjust config.TOOL_RPY_DEG (try +135 / -45 / +45)")
            failures.append("roll-sign")

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("FAILED: " + ", ".join(failures) if failures else "All stages passed.")
    home()
    arm.close()
    return 1 if failures else 0


if __name__ == "__main__":
    _sys.exit(main())
