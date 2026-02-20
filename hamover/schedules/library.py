"""Standard schedule library: Roland-Cerf, constant-speed, and gap-power family."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .protocol import BaseSchedule, validate_schedule


# --- Roland-Cerf ---

@dataclass(slots=True)
class RolandCerfSchedule(BaseSchedule):
    """Roland-Cerf local-adiabatic schedule (paper Eq. 51, runtime Eq. 95)."""

    x: float
    epsilon: float
    T: float = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.x < 1.0):
            raise ValueError(f"x must be in (0,1), got {self.x}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

        self.T = self.runtime(self.x, self.epsilon)
        BaseSchedule.__post_init__(self)
        validate_schedule(self)

    @staticmethod
    def runtime(x: float, epsilon: float) -> float:
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0,1), got {x}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        sqrt_term = np.sqrt(1.0 - x**2)
        return float(np.arctan2(sqrt_term, x) / (epsilon * x * sqrt_term))

    @staticmethod
    def gap_from_s(s: float, x: float) -> float:
        w = x**2
        return float(np.sqrt(max(0.0, 1.0 - 4.0 * s * (1.0 - s) * (1.0 - w))))

    def _theta(self, t: float) -> float:
        tc = self.clamp_time(t)
        x = self.x
        eps = self.epsilon
        sqrt_term = np.sqrt(1.0 - x**2)
        return float(2.0 * eps * x * sqrt_term * tc - np.arctan2(sqrt_term, x))

    def s(self, t: float) -> float:
        x = self.x
        sqrt_term = np.sqrt(1.0 - x**2)
        theta = self._theta(t)
        val = 0.5 + 0.5 * x / sqrt_term * np.tan(theta)
        return float(np.clip(val, 0.0, 1.0))

    def ds_dt(self, t: float) -> float:
        x = self.x
        theta = self._theta(t)
        cos_theta = np.cos(theta)
        return float(self.epsilon * x**2 / (cos_theta * cos_theta))

    def local_gap(self, t: float) -> float:
        return self.gap_from_s(self.s(t), self.x)


# --- Constant-speed ---

@dataclass(slots=True)
class ConstantSpeedSchedule(BaseSchedule):
    """Linear schedule s(t)=t/T (Table II p=0 baseline)."""

    def __post_init__(self) -> None:
        BaseSchedule.__post_init__(self)
        validate_schedule(self)

    def s(self, t: float) -> float:
        return self.clamp_time(t) / self.T

    def ds_dt(self, t: float) -> float:
        _ = t
        return 1.0 / self.T


# --- Gap-power ---

def gap_function(s: float, x: float) -> float:
    """Instantaneous adiabatic search gap Delta_w(s)."""
    w = x**2
    return float(np.sqrt(max(0.0, 1.0 - 4.0 * s * (1.0 - s) * (1.0 - w))))


def table2_properties(p: float) -> tuple[bool, bool]:
    """Return (fixed_point, grover_scaling) according to Table II heuristics."""
    if np.isclose(p, 0.0):
        return False, False
    if np.isclose(p, 1.0):
        return True, False
    if np.isclose(p, 2.0):
        return True, True
    if np.isclose(p, 3.0):
        return True, False

    fixed_point = p >= 1.0
    grover = np.isclose(p, 2.0, atol=0.25)
    return fixed_point, grover


@dataclass(slots=True)
class GapPowerSchedule(BaseSchedule):
    """General gap-power family: ds/dt = epsilon * Delta(s)^p."""

    x: float
    epsilon: float
    p: float
    n_steps: int = 20000

    T: float = field(init=False)
    _s_grid: np.ndarray = field(init=False, repr=False)
    _t_grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.x < 1.0):
            raise ValueError(f"x must be in (0,1), got {self.x}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.n_steps < 500:
            raise ValueError("n_steps must be >= 500 for stable interpolation")

        self._build_interpolation()
        BaseSchedule.__post_init__(self)
        validate_schedule(self)

    def _build_interpolation(self) -> None:
        s_grid = np.linspace(0.0, 1.0, self.n_steps + 1)
        ds = 1.0 / self.n_steps

        gaps = np.array([gap_function(s, self.x) for s in s_grid], dtype=float)
        speed = self.epsilon * np.power(gaps, self.p)

        if np.any(speed <= 0.0):
            raise ValueError("Non-positive speed encountered in gap-power construction")

        dt_ds = 1.0 / speed
        increments = 0.5 * (dt_ds[1:] + dt_ds[:-1]) * ds
        t_grid = np.empty_like(s_grid)
        t_grid[0] = 0.0
        t_grid[1:] = np.cumsum(increments)

        self._s_grid = s_grid
        self._t_grid = t_grid
        self.T = float(t_grid[-1])

    def s(self, t: float) -> float:
        tc = self.clamp_time(t)
        return float(np.interp(tc, self._t_grid, self._s_grid))

    def ds_dt(self, t: float) -> float:
        s_val = self.s(t)
        return float(self.epsilon * gap_function(s_val, self.x) ** self.p)

    def table2_flags(self) -> tuple[bool, bool]:
        return table2_properties(self.p)


__all__ = [
    "ConstantSpeedSchedule",
    "GapPowerSchedule",
    "RolandCerfSchedule",
    "gap_function",
    "table2_properties",
]
