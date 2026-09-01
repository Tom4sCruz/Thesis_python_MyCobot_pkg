"""
Deliberate jitter injection -- the inverse of ``arm._min_jerk()``.

``arm._min_jerk()`` shapes a move so it starts and stops with no velocity or
acceleration step. This module does the opposite on request: it corrupts the
joint setpoints ``Arm._execute()`` / ``Arm._execute_single_joint()`` are about
to stream, so the arm visibly shakes. It exists to compare smooth vs. jerky
motion.

Three independent knobs, set as plain attributes on ``Arm``:

    arm.jerk             Dimensionless roughness dial, 0 = smooth. Drives a
                         per-joint TREMOR (filtered noise added to every
                         setpoint) and an UNEVEN PACE (the commanded speed is
                         modulated up and down), scaled by
                         ``config.JERK_TREMOR_DEG_PER_UNIT`` and
                         ``config.JERK_SPEED_FRAC_PER_UNIT``.
    arm.random_twitch    Probability in [0, 1] that a discrete "flinch" starts
                         on any given streamed setpoint -- so a twitch fires
                         roughly ``random_twitch`` of the time during the move.
    arm.twitch_intensity Peak flinch amplitude, in degrees.

A twitch fires on a random single joint with a random sign, ramps to full
amplitude over ``config.JERK_TWITCH_RISE_TICKS`` control ticks and decays over
``config.JERK_TWITCH_DECAY_TICKS``.

The injector is INERT -- ``offsets()`` returns zeros, ``speed_factor()`` returns
1.0, ``active`` is False -- unless ``jerk > 0`` or both ``random_twitch`` and
``twitch_intensity`` are non-zero. So the default ``Arm`` (all three attributes
0.0) streams exactly the trajectory it always did.

Pure numpy, no hardware, deterministic given the rng. Per tick, call
``offsets()`` first, then ``speed_factor()`` -- both draw from the rng and
advance internal state, so a consistent call order keeps runs reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class _Twitch:
    """One live flinch: a signed peak on a single joint, ramped then decayed."""

    joint: int
    peak_deg: float
    rise: int
    decay: int
    tick: int = 0

    @property
    def done(self) -> bool:
        return self.tick >= self.rise + self.decay

    def value(self) -> float:
        """This twitch's contribution (deg) for the current tick."""
        if self.tick < self.rise:
            frac = (self.tick + 1) / max(self.rise, 1)
        else:
            frac = 1.0 - (self.tick - self.rise + 1) / max(self.decay, 1)
        return self.peak_deg * max(frac, 0.0)

    def advance(self) -> None:
        self.tick += 1


class JerkInjector:
    """Deliberate jitter added to joint setpoints (see module docstring)."""

    def __init__(self, jerk, random_twitch, twitch_intensity, dof, rng):
        self.jerk = max(float(jerk), 0.0)
        self.random_twitch = min(max(float(random_twitch), 0.0), 1.0)  # probability
        self.twitch_intensity = float(twitch_intensity)
        self.dof = int(dof)
        self.rng = rng

        self._tremor_sigma = config.JERK_TREMOR_DEG_PER_UNIT * self.jerk
        self._pace_sigma = config.JERK_SPEED_FRAC_PER_UNIT * self.jerk
        self._rho = float(config.JERK_TREMOR_CORRELATION)
        self._innov = float(np.sqrt(max(1.0 - self._rho ** 2, 0.0)))
        self._cap = float(config.JERK_MAX_DEG)

        self._tremor = np.zeros(self.dof)   # AR(1) state, per joint
        self._pace = 0.0                    # AR(1) state, scalar
        self._twitches: list[_Twitch] = []

    @property
    def active(self) -> bool:
        return self.jerk > 0.0 or (
            self.random_twitch > 0.0 and self.twitch_intensity != 0.0
        )

    def reset(self) -> None:
        """Zero the AR(1) state and drop every live twitch."""
        self._tremor[:] = 0.0
        self._pace = 0.0
        self._twitches.clear()

    def offsets(self, dt: float) -> np.ndarray:
        """
        (DOF,) degrees to ADD to this tick's setpoint. Advances internal state.

        ``dt`` (control period, seconds) is accepted for call-signature
        symmetry with ``speed_factor()`` but is not currently used -- the
        twitch chance is a flat per-call probability (``random_twitch``).
        """
        if not self.active:
            return np.zeros(self.dof)

        # -- tremor: per-joint AR(1)-filtered gaussian ---------------------
        if self._tremor_sigma > 0.0:
            self._tremor = (
                self._rho * self._tremor
                + self._innov * self._tremor_sigma * self.rng.standard_normal(self.dof)
            )
        out = np.array(self._tremor, dtype=float)

        # -- twitches: maybe start one, then sum the live ones ------------
        if self.random_twitch > 0.0 and self.twitch_intensity != 0.0:
            if self.rng.random() < self.random_twitch:
                sign = 1.0 if self.rng.random() < 0.5 else -1.0
                # 50-100% of twitch_intensity, so flinches vary in size
                amp = sign * abs(self.twitch_intensity) * (0.5 + 0.5 * self.rng.random())
                self._twitches.append(_Twitch(
                    joint=int(self.rng.integers(self.dof)),
                    peak_deg=amp,
                    rise=int(config.JERK_TWITCH_RISE_TICKS),
                    decay=int(config.JERK_TWITCH_DECAY_TICKS),
                ))
            live: list[_Twitch] = []
            for tw in self._twitches:
                out[tw.joint] += tw.value()
                tw.advance()
                if not tw.done:
                    live.append(tw)
            self._twitches = live

        return np.clip(out, -self._cap, self._cap)

    def speed_factor(self) -> float:
        """Multiplier for this tick's commanded joint speed ('uneven pace')."""
        if not self.active or self._pace_sigma <= 0.0:
            return 1.0
        self._pace = (
            self._rho * self._pace
            + self._innov * self._pace_sigma * float(self.rng.standard_normal())
        )
        return float(np.clip(
            1.0 + self._pace,
            config.JERK_SPEED_FACTOR_MIN,
            config.JERK_SPEED_FACTOR_MAX,
        ))
