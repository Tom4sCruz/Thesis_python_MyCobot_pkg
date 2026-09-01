"""
Thin wrapper around pymycobot.

Everything that touches the serial port goes through here, for three reasons:

1. pymycobot is not thread safe and the UART is a single shared resource, so
   every call is serialised behind one lock.

2. The firmware sporadically returns [] or a short list from get_angles(). Left
   unhandled, that None propagates into the kinematics and produces a confusing
   crash far from the cause. Here it is retried.

3. pymycobot's send_angles(angles, speed) takes an INTEGER speed 0-100 in
   arbitrary units -- there is no deg/s field anywhere in the serial protocol.
   This module owns the float-deg/s -> firmware-integer conversion and reports
   the quantisation error rather than hiding it.

Anything this module does not wrap is still reachable through `.raw`, which is
the live pymycobot object, under the same lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import config
from .kinematics import check_joint_limits

log = logging.getLogger(__name__)


class ArmError(RuntimeError):
    pass


@dataclass
class SpeedConversion:
    """What you asked for, what the firmware actually got, and the difference."""
    requested_dps: float
    firmware_speed: int
    actual_dps: float

    @property
    def error_dps(self) -> float:
        return self.actual_dps - self.requested_dps


def dps_to_firmware_speed(deg_per_s: float) -> SpeedConversion:
    """Map a float deg/s onto pymycobot's integer 1-100 speed scale."""
    if deg_per_s <= 0:
        raise ValueError("angular velocity must be positive")
    raw = 100.0 * deg_per_s / config.DEG_PER_S_AT_SPEED_100
    speed = int(round(np.clip(raw, config.FIRMWARE_SPEED_MIN, config.FIRMWARE_SPEED_MAX)))
    actual = speed / 100.0 * config.DEG_PER_S_AT_SPEED_100
    return SpeedConversion(deg_per_s, speed, actual)


class ArmConnection:
    """Owns the pymycobot object and serialises access to it."""

    def __init__(
        self,
        port: str = config.DEFAULT_PORT,
        baudrate: int = config.DEFAULT_BAUDRATE,
        mock: bool = False,
    ):
        self._lock = threading.RLock()
        self.mock = mock
        self._single_joint = 0

        if mock:
            from .mock import MockMyCobot
            self._mc = MockMyCobot()
            log.warning("ArmConnection is in MOCK mode -- no hardware will move.")
        else:
            try:
                from pymycobot import MyCobot280
            except ImportError:
                try:
                    from pymycobot.mycobot import MyCobot as MyCobot280  # older versions
                except ImportError as exc:
                    raise ArmError(
                        "pymycobot is not installed. `pip install pymycobot`, or "
                        "construct ArmConnection(mock=True) to work offline."
                    ) from exc
            self._mc = MyCobot280(port, baudrate)
            time.sleep(0.1)

        try:
            self.set_fresh_mode(1)
        except Exception as exc:
            log.warning("could not set fresh_mode: %s", exc)

    # -- escape hatch -------------------------------------------------------

    @property
    def raw(self):
        """The live pymycobot object, for anything not wrapped here."""
        return self._mc

    @property
    def lock(self) -> threading.RLock:
        """Hold this when using .raw, so you don't collide with a running move."""
        return self._lock

    # -- state --------------------------------------------------------------

    def is_power_on(self) -> bool:
        with self._lock:
            try:
                return bool(self._mc.is_power_on())
            except Exception as exc:
                log.debug("is_power_on failed: %s", exc)
                return False

    def power_on(self):
        with self._lock:
            return self._mc.power_on()

    def set_fresh_mode(self, mode: int) -> int:
        """
        mode=1 ("refresh"): a newly received command immediately overrides
        whatever the arm was doing toward the previous target -- pymycobot's
        own docs describe it as "always execute the latest command first".
        mode=0 ("interpolation"): commands queue and run strictly in sequence.

        Set to 1 at connection time, not for smoothness -- fresh_mode does NOT
        blend velocity between commands, it just preempts. It matters here
        because Arm._execute() streams setpoints faster than the arm can fully
        settle between them and assumes each new one immediately supersedes
        the last (see cobotmotion's earlier "PWM-style hop-then-hold" model).
        Under mode 0 that assumption could silently fail -- commands would
        queue into a backlog instead of tracking the schedule -- and mock mode
        can't reveal that, since it applies every command instantly.
        """
        with self._lock:
            return self._mc.set_fresh_mode(mode)


    def get_fresh_mode(self) -> int:
        with self._lock:
            return self._mc.get_fresh_mode()

    def set_single_joint(self, mode: int) -> int:
        """
        Enable or disable single-joint path execution.

        mode=1: one joint moves at a time
        mode=0: normal multi-joint execution
        """
        if mode not in (0, 1):
            raise ValueError("single_joint mode must be 0 or 1")

        with self._lock:
            self._single_joint = mode

        return mode

    def get_single_joint(self) -> int:
        with self._lock:
            return self._single_joint


    def get_angles(self, retries: int = 4) -> list[float]:
        """Joint angles in degrees. Retries past the firmware's empty replies.

        The firmware sometimes answers a read with a bare status int (e.g. -1)
        instead of a list -- often when the serial reply for a previous command
        (a gripper move, say) has not been fully drained. That is not sized, so
        it must be rejected before len(), or it raises TypeError instead of
        being retried.
        """
        with self._lock:
            for attempt in range(retries):
                try:
                    angles = self._mc.get_angles()
                except Exception as exc:
                    log.debug("get_angles raised (attempt %d): %s", attempt, exc)
                    angles = None
                if (isinstance(angles, (list, tuple))
                        and len(angles) == config.DOF
                        and all(a is not None for a in angles)):
                    return [float(a) for a in angles]
                log.debug("get_angles bad reply (attempt %d): %r", attempt, angles)
                time.sleep(0.05)
        raise ArmError(f"could not read joint angles after {retries} attempts")

    def get_coords_firmware(self) -> list[float]:
        """
        The FIRMWARE's own FK, in mm/deg. Used only by verify_fk.py to validate
        our DH table -- never in the control path.
        """
        with self._lock:
            for _ in range(4):
                try:
                    c = self._mc.get_coords()
                except Exception:
                    c = None
                if isinstance(c, (list, tuple)) and len(c) == 6:
                    return [float(v) for v in c]
                time.sleep(0.05)
        raise ArmError("could not read coords from firmware")

    # -- motion -------------------------------------------------------------

    def send_angles(self, angles_deg, deg_per_s: float) -> SpeedConversion:
        """
        Command all six joints. Refuses out-of-limit angles rather than letting
        the firmware clamp them silently.
        """
        angles = np.asarray(angles_deg, dtype=float)
        if angles.shape != (config.DOF,):
            raise ValueError(f"expected {config.DOF} angles, got {angles.shape}")

        problems = check_joint_limits(angles)
        if problems:
            raise ArmError("refusing out-of-limit angles: " + "; ".join(problems))

        conv = dps_to_firmware_speed(deg_per_s)
        # The wire protocol quantises to 0.01 deg anyway; round here so logs
        # match what the arm actually received.
        payload = [round(float(a), 2) for a in angles]
        with self._lock:
            self._mc.send_angles(payload, conv.firmware_speed)
        return conv

    def send_angle(self, joint_id: int, angle_deg: float, deg_per_s: float) -> SpeedConversion:
        """Command a single joint. joint_id is 1-based, matching pymycobot."""
        if not 1 <= joint_id <= config.DOF:
            raise ValueError(f"joint_id must be 1..{config.DOF}, got {joint_id}")
        lo, hi = config.joint_limits_array()[joint_id - 1]
        if not lo <= angle_deg <= hi:
            raise ArmError(
                f"J{joint_id}={angle_deg:.2f} outside soft limits [{lo:.1f}, {hi:.1f}]"
            )
        conv = dps_to_firmware_speed(deg_per_s)
        with self._lock:
            self._mc.send_angle(joint_id, round(float(angle_deg), 2), conv.firmware_speed)
        return conv

    def stop(self):
        with self._lock:
            try:
                return self._mc.stop()
            except Exception as exc:
                log.error("stop() failed: %s", exc)
                return 0

    def release_all_servos(self):
        with self._lock:
            return self._mc.release_all_servos()

    # -- gripper ----------------------------------------------------------------

    def set_gripper_value(
        self,
        value: int,
        speed: int = config.GRIPPER_DEFAULT_SPEED,
        gripper_type: int | None = None,
    ):
        """
        Set the gripper opening: `value` 0-100 (0 closed, 100 open), `speed` an
        integer 1-100. Thin wrapper over pymycobot's set_gripper_value under the
        shared serial lock. The return value is whatever pymycobot returns
        (often None on real hardware) -- success is the absence of an exception,
        not a truthy result.
        """
        value = int(value)
        if not 0 <= value <= 100:
            raise ValueError(f"gripper value must be 0-100, got {value}")
        speed = int(speed)
        if not config.FIRMWARE_SPEED_MIN <= speed <= config.FIRMWARE_SPEED_MAX:
            raise ValueError(
                f"gripper speed must be {config.FIRMWARE_SPEED_MIN}-"
                f"{config.FIRMWARE_SPEED_MAX}, got {speed}"
            )
        with self._lock:
            if gripper_type is None:
                return self._mc.set_gripper_value(value, speed)
            return self._mc.set_gripper_value(value, speed, gripper_type)

    def is_gripper_moving(self) -> bool:
        with self._lock:
            try:
                return bool(self._mc.is_gripper_moving())
            except Exception as exc:
                log.debug("is_gripper_moving failed: %s", exc)
                return False

    def close(self):
        try:
            self.stop()
        finally:
            serial = getattr(self._mc, "_serial_port", None)
            if serial is not None and hasattr(serial, "close"):
                try:
                    serial.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
