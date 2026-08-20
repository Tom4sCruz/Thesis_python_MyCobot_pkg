#!/usr/bin/env python3
"""
STEP 1 -- run this before trusting any Cartesian number.

Compares this package's forward kinematics against the FIRMWARE's own
get_coords() at several poses. If the residual is more than a few millimetres,
the DH table in config.py does not describe your arm, and every IK result you
get will be quietly wrong.

Also settles two things you cannot guess:
  * which Euler convention the firmware uses for rx/ry/rz
  * whether JOINT_1_HEIGHT_CM is being double-counted in z

    python3 scripts/verify_fk.py --port /dev/ttyTHS1
    python3 scripts/verify_fk.py --mock     # flow check only; FK vs FK is circular
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time

import numpy as np

from armik import config, kinematics
from armik.connection import ArmConnection

TEST_POSES = [
    [0, 0, -90, 0, 0, 0],
    [20, -20, -70, -10, 15, 0],
    [-30, 10, -60, -30, -20, 25],
    [45, -40, -50, 0, 40, -30],
    [0, -30, -30, -30, 0, 0],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--speed", type=float, default=30.0, help="deg/s for setup moves")
    ap.add_argument("--settle", type=float, default=4.0, help="seconds to settle")
    args = ap.parse_args()

    if args.mock:
        print("NOTE: in mock mode the 'firmware' FK IS our FK, so the residual")
        print("      is trivially zero. This only checks the script runs.\n")

    conn = ArmConnection(args.port, args.baud, mock=args.mock)
    if not conn.is_power_on():
        conn.power_on()
        time.sleep(1.0)

    rows = []
    for pose in TEST_POSES:
        print(f"\n--> moving to {pose}")
        conn.send_angles(pose, args.speed)
        time.sleep(args.settle)

        q = conn.get_angles()
        fw = conn.get_coords_firmware()
        ours = kinematics.pose_coords(q)

        pos_err = float(np.linalg.norm(np.array(fw[:3]) - ours[:3]))
        print(f"    joints   : {[round(v, 2) for v in q]}")
        print(f"    firmware : {[round(v, 2) for v in fw]}")
        print(f"    ours     : {[round(float(v), 2) for v in ours]}")
        print(f"    pos error: {pos_err:.2f} mm")
        rows.append((q, fw, pos_err))

    errs = np.array([r[2] for r in rows])
    print("\n" + "=" * 66)
    print(f"POSITION RESIDUAL   mean {errs.mean():.2f} mm   max {errs.max():.2f} mm")
    if errs.max() < 3.0:
        print("VERDICT: DH table looks correct. Proceed.")
    elif errs.max() < 15.0:
        print("VERDICT: close but off. Check the d/a values, and whether a")
        print("         gripper is mounted -- config.py assumes a bare flange.")
    else:
        print("VERDICT: DH table does NOT match this robot. Do not use IK until")
        print("         this is fixed; pull link lengths from the mycobot_280_jn URDF.")

    print("\nEULER CONVENTION (orientation residual, degrees):")
    for conv in ("xyz_fixed", "zyx_fixed"):
        res = []
        for q, fw, _ in rows:
            R_ours = kinematics.forward_kinematics(q)[:3, :3]
            R_fw = kinematics.rpy_to_matrix(fw[3], fw[4], fw[5], convention=conv)
            dR = R_fw @ R_ours.T
            ang = np.rad2deg(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            res.append(ang)
        mark = " <-- current" if conv == config.EULER_CONVENTION else ""
        print(f"    {conv:<12} mean {np.mean(res):7.2f}   max {np.max(res):7.2f}{mark}")
    print("    Set config.EULER_CONVENTION to whichever is smaller.")

    print("\nBASE HEIGHT / z ORIGIN:")
    z_dh = config.DH_TABLE[0][1] / 10.0
    print(f"    DH d1              = {z_dh:.3f} cm")
    print(f"    JOINT_1_HEIGHT_CM  = {config.JOINT_1_HEIGHT_CM:.3f} cm")
    print(f"    Z_RELATIVE_TO_JOINT1 = {config.Z_RELATIVE_TO_JOINT1}")
    zs_fw = np.array([r[1][2] for r in rows]) / 10.0
    zs_ours = np.array([kinematics.pose_coords(r[0])[2] for r in rows]) / 10.0
    off = float(np.mean(zs_fw - zs_ours))
    print(f"    mean (firmware z - our z) = {off:+.3f} cm")
    if abs(off - config.JOINT_1_HEIGHT_CM) < 2.0:
        print("    !! That offset is close to the base height -- you are probably")
        print("       double-counting it. Check Z_RELATIVE_TO_JOINT1.")
    elif abs(off) < 1.0:
        print("    OK: both measure z from the same origin (the table).")

    conn.close()


if __name__ == "__main__":
    main()
