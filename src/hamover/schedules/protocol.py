"""Base schedule protocol, validation, and common runtime utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Schedule(Protocol):
    """Protocol for adiabatic schedules s:[0,T]->[0,1]."""

    T: float

    def s(self, t: float) -> float:
        ...

    def ds_dt(self, t: float) -> float:
        ...


@dataclass(slots=True)
class BaseSchedule:
    """Base utility class for common schedule validation/clamping behavior."""

    T: float

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")

    def clamp_time(self, t: float) -> float:
        return float(np.clip(t, 0.0, self.T))


def validate_schedule(schedule: Schedule, n_points: int = 1001, tol: float = 1e-8) -> None:
    """Validate schedule boundary and monotonicity conditions."""
    if schedule.T <= 0.0:
        raise ValueError(f"Schedule runtime T must be positive, got {schedule.T}")

    s0 = float(schedule.s(0.0))
    sT = float(schedule.s(schedule.T))
    if not np.isclose(s0, 0.0, atol=tol, rtol=0.0):
        raise ValueError(f"Boundary condition failed: s(0)={s0}")
    if not np.isclose(sT, 1.0, atol=tol, rtol=0.0):
        raise ValueError(f"Boundary condition failed: s(T)={sT}")

    grid = np.linspace(0.0, schedule.T, n_points)
    vals = np.array([float(schedule.s(t)) for t in grid], dtype=float)

    if np.any(vals < -tol) or np.any(vals > 1.0 + tol):
        raise ValueError("Schedule leaves [0,1] range")

    diffs = np.diff(vals)
    if np.any(diffs < -tol):
        raise ValueError("Schedule is not monotonic non-decreasing")


__all__ = ["BaseSchedule", "Schedule", "validate_schedule"]
