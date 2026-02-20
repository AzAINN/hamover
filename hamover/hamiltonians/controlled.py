"""Control-field Hamiltonians: su(2) universal representation and monotonic scenarios."""

from __future__ import annotations

from typing import Callable

import numpy as np

from hamover.core.types import SU2Fields


class SU2Hamiltonian:
    """General su(2) Hamiltonian H(t) = [[Omega, omega], [omega*, -Omega]]."""

    def __init__(
        self,
        omega_func: Callable[[float], complex],
        Omega_func: Callable[[float], float],
        hbar: float = 1.0,
        T: float | None = None,
    ) -> None:
        if hbar <= 0.0:
            raise ValueError(f"hbar must be positive, got {hbar}")
        self._omega = omega_func
        self._Omega = Omega_func
        self.hbar = float(hbar)
        self.T = T

    @property
    def is_time_dependent(self) -> bool:
        # Generic callable fields are assumed time-dependent.
        return True

    def omega(self, t: float) -> complex:
        return complex(self._omega(t))

    def Omega(self, t: float) -> float:
        return float(self._Omega(t))

    def matrix(self, t: float) -> np.ndarray:
        w = self.omega(t)
        O = self.Omega(t)
        return np.array([[O, w], [np.conj(w), -O]], dtype=complex)

    def energy_gap(self, t: float) -> float:
        vals = np.linalg.eigvalsh(self.matrix(t))
        return float(vals[1] - vals[0])

    def to_su2(self) -> SU2Fields:
        return SU2Fields(omega=self.omega, Omega=self.Omega, T=self.T)

    def is_static(self, t0: float = 0.0, t1: float = 1.0, atol: float = 1e-12) -> bool:
        """Numerical check for static fields; useful for GQS/Farhi-Gutmann mapping checks."""
        return (
            np.isclose(self.omega(t0), self.omega(t1), atol=atol, rtol=0.0)
            and np.isclose(self.Omega(t0), self.Omega(t1), atol=atol, rtol=0.0)
        )

    @classmethod
    def from_adiabatic(
        cls,
        schedule: object,
        x: float,
        hbar: float = 1.0,
    ) -> "SU2Hamiltonian":
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")

        def omega(t: float) -> complex:
            s = float(schedule.s(t))
            return complex(-(1.0 - s) * x * np.sqrt(1.0 - x**2))

        def Omega(t: float) -> float:
            s = float(schedule.s(t))
            return float(0.5 * ((1.0 - s) * (1.0 - 2.0 * x**2) - s))

        T = float(schedule.T) if hasattr(schedule, "T") else None
        return cls(omega_func=omega, Omega_func=Omega, hbar=hbar, T=T)

    @classmethod
    def from_nonadiabatic(
        cls,
        f_func: Callable[[float], float],
        g_func: Callable[[float], float],
        x: float,
        hbar: float = 1.0,
        T: float | None = None,
    ) -> "SU2Hamiltonian":
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")

        def omega(t: float) -> complex:
            return complex(-float(f_func(t)) * x * np.sqrt(1.0 - x**2))

        def Omega(t: float) -> float:
            f = float(f_func(t))
            g = float(g_func(t))
            return float(0.5 * (f * (1.0 - 2.0 * x**2) - g))

        return cls(omega_func=omega, Omega_func=Omega, hbar=hbar, T=T)

    def to_schedule(self, x: float) -> Callable[[float], float]:
        """Recover adiabatic schedule via the Eq. 92 ratio structure."""
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")

        def s(t: float) -> float:
            w = self.omega(t)
            O = self.Omega(t)
            if abs(O) < 1e-15:
                return 0.5

            if abs(np.imag(w)) > 1e-12:
                raise ValueError(
                    "Complex omega cannot be inverted to adiabatic s(t) with Eq. 92 real-ratio form"
                )

            ratio = float(np.real(w) / O)
            num = 2.0 * x * np.sqrt(1.0 - x**2) + ratio * (1.0 - 2.0 * x**2)
            den = 2.0 * x * np.sqrt(1.0 - x**2) + 2.0 * ratio * (1.0 - x**2)
            if abs(den) < 1e-15:
                return 0.5
            return float(np.clip(num / den, 0.0, 1.0))

        return s

    def verify_adiabatic_mapping(
        self,
        schedule: object,
        x: float,
        n_points: int = 101,
    ) -> float:
        """Return max |s_recovered(t)-s_original(t)| over [0,T] when Eq. 92 inversion is valid."""
        if not hasattr(schedule, "T"):
            raise ValueError("schedule must define T")
        s_back = self.to_schedule(x)
        grid = np.linspace(0.0, float(schedule.T), n_points)
        errors = [abs(float(schedule.s(t)) - s_back(t)) for t in grid]
        return float(max(errors))


class MonotonicScenario1:
    """Exactly-solvable monotonic model, scenario 1 (paper Eqs. 64-77)."""

    def __init__(self, omega_0: float, xi: float, c: float, hbar: float = 1.0) -> None:
        if omega_0 <= 0.0:
            raise ValueError(f"omega_0 must be positive, got {omega_0}")
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi}")
        if c == 0.0:
            raise ValueError("c must be nonzero")
        if hbar <= 0.0:
            raise ValueError(f"hbar must be positive, got {hbar}")

        self.omega_0 = float(omega_0)
        self.xi = float(xi)
        self.c = float(c)
        self.hbar = float(hbar)

    @classmethod
    def monotonic(cls, omega_0: float, c: float, hbar: float = 1.0) -> "MonotonicScenario1":
        """Set xi to n=0 monotonic branch value (Eq. 75)."""
        if c == 0.0:
            raise ValueError("c must be nonzero")
        xi = (2.0 / np.pi) * np.sqrt(hbar**2 + c**2) / c * omega_0 / hbar
        return cls(omega_0=omega_0, xi=float(abs(xi)), c=c, hbar=hbar)

    def phase_argument(self, t: float) -> float:
        h = self.hbar
        c = self.c
        w0 = self.omega_0
        xi = self.xi
        return float(np.sqrt(h**2 + c**2) / (h * c) * (w0 / xi) * (1.0 - np.exp(-xi * t)))

    def alpha_squared(self, t: float) -> float:
        Phi = self.phase_argument(t)
        h = self.hbar
        c = self.c
        return float((h**2 + c**2 * np.cos(Phi) ** 2) / (h**2 + c**2))

    def beta_squared(self, t: float) -> float:
        return float(1.0 - self.alpha_squared(t))

    def transition_probability(self, t: float, x: float) -> float:
        """Eq. 71 simplified in the resonant-style approximation used in docs."""
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")
        a2 = self.alpha_squared(t)
        b2 = self.beta_squared(t)
        return float(a2 * x**2 + b2 * (1.0 - x**2))

    def figure_data(self, t_max: float, n_points: int, x: float) -> tuple[np.ndarray, np.ndarray]:
        """Convenience helper for Fig. 3-style reproduction."""
        if t_max <= 0.0:
            raise ValueError(f"t_max must be positive, got {t_max}")
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        ts = np.linspace(0.0, t_max, n_points)
        probs = np.array([self.transition_probability(t, x) for t in ts], dtype=float)
        return ts, probs


class MonotonicScenario2:
    """Exactly-solvable monotonic model, scenario 2 (paper Eqs. 78-89)."""

    def __init__(self, omega_0: float, xi: float, hbar: float = 1.0) -> None:
        if omega_0 <= 0.0:
            raise ValueError(f"omega_0 must be positive, got {omega_0}")
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi}")
        if hbar <= 0.0:
            raise ValueError(f"hbar must be positive, got {hbar}")

        self.omega_0 = float(omega_0)
        self.xi = float(xi)
        self.hbar = float(hbar)

    @classmethod
    def monotonic(cls, omega_0: float, hbar: float = 1.0) -> "MonotonicScenario2":
        """Set xi to strict monotonic value in Eq. 89."""
        xi = (4.0 / np.pi) * omega_0 / hbar * (np.pi / 4.0 - 0.5)
        return cls(omega_0=omega_0, xi=float(xi), hbar=hbar)

    def _F(self, t: float) -> float:
        w0 = self.omega_0
        h = self.hbar
        xi = self.xi
        exp_t = np.exp(xi * t)
        return float(
            (2.0 * w0)
            / (h * xi)
            * (exp_t / (1.0 + np.exp(2.0 * xi * t)) - np.arctan(np.exp(-xi * t)) + np.pi / 4.0 - 0.5)
        )

    def alpha_squared(self, t: float) -> float:
        return float(np.cos(self._F(t)) ** 2)

    def beta_squared(self, t: float) -> float:
        return float(np.sin(self._F(t)) ** 2)

    def transition_probability(self, t: float, x: float) -> float:
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")
        a2 = self.alpha_squared(t)
        b2 = self.beta_squared(t)
        return float(a2 * x**2 + b2 * (1.0 - x**2))

    def figure_data(self, t_max: float, n_points: int, x: float) -> tuple[np.ndarray, np.ndarray]:
        """Convenience helper for Fig. 4-style reproduction."""
        if t_max <= 0.0:
            raise ValueError(f"t_max must be positive, got {t_max}")
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        ts = np.linspace(0.0, t_max, n_points)
        probs = np.array([self.transition_probability(t, x) for t in ts], dtype=float)
        return ts, probs


__all__ = ["MonotonicScenario1", "MonotonicScenario2", "SU2Hamiltonian"]
