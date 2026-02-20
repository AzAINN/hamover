"""Schedule-driven Hamiltonians (adiabatic and nonadiabatic)."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from hamover.core.types import SU2Fields


class AdiabaticSchedule(Protocol):
    T: float

    def s(self, t: float) -> float:
        ...

    def ds_dt(self, t: float) -> float:
        ...


class AdiabaticSearchHamiltonian:
    """Adiabatic search Hamiltonian H(t) = [1-s(t)]H0 + s(t)H1."""

    def __init__(self, schedule: AdiabaticSchedule, x: float, hbar: float = 1.0) -> None:
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")
        if hbar <= 0.0:
            raise ValueError(f"hbar must be positive, got {hbar}")

        self.schedule = schedule
        self.x = float(x)
        self.hbar = float(hbar)

    @property
    def is_time_dependent(self) -> bool:
        return True

    def matrix(self, t: float) -> np.ndarray:
        s = float(self.schedule.s(t))
        x = self.x

        h00 = (1.0 - s) * (1.0 - x**2)
        h01 = -(1.0 - s) * x * np.sqrt(1.0 - x**2)
        h11 = (1.0 - s) * x**2 + s

        return np.array([[h00, h01], [h01, h11]], dtype=complex)

    def instantaneous_gap(self, t: float) -> float:
        s = float(self.schedule.s(t))
        w = self.x**2
        return float(np.sqrt(max(0.0, 1.0 - 4.0 * s * (1.0 - s) * (1.0 - w))))

    def energy_gap(self, t: float) -> float:
        return self.instantaneous_gap(t)

    def to_su2(self) -> SU2Fields:
        x = self.x

        def omega(t: float) -> complex:
            s = float(self.schedule.s(t))
            return complex(-(1.0 - s) * x * np.sqrt(1.0 - x**2))

        def Omega(t: float) -> float:
            s = float(self.schedule.s(t))
            return float(0.5 * ((1.0 - s) * (1.0 - 2.0 * x**2) - s))

        T = float(self.schedule.T) if hasattr(self.schedule, "T") else None
        return SU2Fields(omega=omega, Omega=Omega, T=T)

    def check_adiabatic_condition(self, t: float, epsilon: float) -> bool:
        """Simplified local adiabatic check used throughout the docs: |ds/dt| <= eps*Delta^2."""
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        gap = self.instantaneous_gap(t)
        ds_dt = float(self.schedule.ds_dt(t))
        return abs(ds_dt) <= epsilon * gap**2

    def verify_su2_schedule_recovery(self, n_points: int = 101) -> float:
        """Return max schedule reconstruction error using SU2 Eq. 92-style inversion."""
        from .controlled import SU2Hamiltonian

        su2 = SU2Hamiltonian.from_adiabatic(self.schedule, self.x, hbar=self.hbar)
        return su2.verify_adiabatic_mapping(self.schedule, self.x, n_points=n_points)


def roland_cerf_runtime(x: float, epsilon: float) -> float:
    """Eq. 95 runtime used to verify Grover O(1/x)=O(sqrt(N)) scaling."""
    if not (0.0 < x < 1.0):
        raise ValueError(f"x must be in (0,1), got {x}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    sqrt_term = np.sqrt(1.0 - x**2)
    return float(np.arctan2(sqrt_term, x) / (epsilon * x * sqrt_term))


def estimate_grover_slope(x_values: Sequence[float], epsilon: float) -> float:
    """Fit log(T_RC) vs log(1/x). Slope near 1 indicates Grover scaling."""
    if len(x_values) < 2:
        raise ValueError("Need at least two x samples")
    x = np.asarray(x_values, dtype=float)
    if np.any(x <= 0.0) or np.any(x >= 1.0):
        raise ValueError("x samples must lie in (0,1)")
    T = np.array([roland_cerf_runtime(float(v), epsilon) for v in x], dtype=float)
    slope, _ = np.polyfit(np.log(1.0 / x), np.log(T), 1)
    return float(slope)


class NonadiabaticSearchHamiltonian:
    """Nonadiabatic search Hamiltonian H(t) = f(t)H0 + g(t)H1."""

    def __init__(
        self,
        f_func: Callable[[float], float],
        g_func: Callable[[float], float],
        x: float,
        hbar: float = 1.0,
    ) -> None:
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")
        if hbar <= 0.0:
            raise ValueError(f"hbar must be positive, got {hbar}")

        self.f = f_func
        self.g = g_func
        self.x = float(x)
        self.hbar = float(hbar)

    @property
    def is_time_dependent(self) -> bool:
        return True

    def matrix(self, t: float) -> np.ndarray:
        f = float(self.f(t))
        g = float(self.g(t))
        x = self.x

        h00 = f * (1.0 - x**2)
        h01 = -f * x * np.sqrt(1.0 - x**2)
        h11 = f * x**2 + g

        return np.array([[h00, h01], [h01, h11]], dtype=complex)

    def energy_gap(self, t: float) -> float:
        vals = np.linalg.eigvalsh(self.matrix(t))
        return float(vals[1] - vals[0])

    def to_su2(self) -> SU2Fields:
        def omega(t: float) -> complex:
            return complex(self.matrix(t)[0, 1])

        def Omega(t: float) -> float:
            H = self.matrix(t)
            return float(0.5 * np.real(H[0, 0] - H[1, 1]))

        return SU2Fields(omega=omega, Omega=Omega, T=None)


__all__ = [
    "AdiabaticSearchHamiltonian",
    "NonadiabaticSearchHamiltonian",
    "estimate_grover_slope",
    "roland_cerf_runtime",
]
