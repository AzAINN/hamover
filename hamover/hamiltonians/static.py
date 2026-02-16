"""Static (time-independent) Hamiltonians: GQS, Farhi-Gutmann, and Rabi."""

from __future__ import annotations

import numpy as np

from hamover.core.constants import BOHR_MAGNETON, overlap_from_fields
from hamover.core.types import SU2Fields


class GQSHamiltonian:
    """Generalized quantum search Hamiltonian (paper Section II, Eq. 1)."""

    def __init__(
        self,
        alpha: float,
        beta: complex,
        delta: float,
        x: float,
        E: float,
        hbar: float = 1.0,
    ) -> None:
        if not (0.0 < x < 1.0):
            raise ValueError(f"x must be in (0, 1), got {x}")
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if hbar <= 0.0:
            raise ValueError(f"hbar must be positive, got {hbar}")

        self.alpha = float(alpha)
        self.beta = complex(beta)
        self.gamma = np.conj(self.beta)
        self.delta = float(delta)
        self.x = float(x)
        self.E = float(E)
        self.hbar = float(hbar)

        sqrt_term = np.sqrt(1.0 - self.x**2)
        self.h11 = self.E * (
            self.alpha + (self.beta + self.gamma) * self.x + self.delta * self.x**2
        )
        self.h12 = self.E * sqrt_term * (self.beta + self.delta * self.x)
        self.h21 = self.E * sqrt_term * (self.gamma + self.delta * self.x)
        self.h22 = self.E * self.delta * (1.0 - self.x**2)

    @property
    def is_time_dependent(self) -> bool:
        return False

    def matrix(self, t: float = 0.0) -> np.ndarray:
        _ = t
        return np.array([[self.h11, self.h12], [self.h21, self.h22]], dtype=complex)

    def eigenvalues(self) -> tuple[float, float]:
        vals = np.linalg.eigvalsh(self.matrix(0.0))
        return float(vals[0]), float(vals[1])

    def energy_gap(self, t: float = 0.0) -> float:
        vals = np.linalg.eigvalsh(self.matrix(t))
        return float(vals[1] - vals[0])

    def t_star(self) -> float:
        gap = self.energy_gap(0.0)
        if gap <= 0.0:
            raise ValueError(f"Energy gap must be positive, got {gap}")
        return float(np.pi * self.hbar / gap)

    def P_max(self) -> float:
        h11, h12, h21, h22 = self.h11, self.h12, self.h21, self.h22
        den = h12 * h21 / self.hbar**2 + (h11 - h22) ** 2 / (4.0 * self.hbar**2)
        if abs(den) < 1e-16:
            raise ValueError("Degenerate denominator in P_max expression")

        num = (
            0.5 * (h11 - h22) / self.hbar * self.x
            + h12 / self.hbar * np.sqrt(1.0 - self.x**2)
        ) ** 2
        return float(np.real(num / den))

    def transition_probability(self, t: float) -> float:
        h11, h12, h21, h22 = self.h11, self.h12, self.h21, self.h22
        omega = np.sqrt(h12 * h21 / self.hbar**2 + (h11 - h22) ** 2 / (4.0 * self.hbar**2))

        den = h12 * h21 / self.hbar**2 + (h11 - h22) ** 2 / (4.0 * self.hbar**2)
        if abs(den) < 1e-16:
            raise ValueError("Degenerate denominator in transition expression")

        num = (
            0.5 * (h11 - h22) / self.hbar * self.x
            + h12 / self.hbar * np.sqrt(1.0 - self.x**2)
        ) ** 2

        cos_term = self.x**2 * np.cos(omega * t) ** 2
        sin_term = np.real(num / den) * np.sin(omega * t) ** 2
        return float(np.real(cos_term + sin_term))

    def to_su2(self) -> SU2Fields:
        trace_half = 0.5 * (self.h11 + self.h22)
        Omega0 = 0.5 * np.real(self.h11 - self.h22)
        omega0 = self.h12

        _ = trace_half  # dropped global phase contribution
        return SU2Fields(
            omega=lambda t: omega0,
            Omega=lambda t: float(Omega0),
            T=None,
        )

    def verify_su2_mapping(self, atol: float = 1e-10) -> bool:
        """Check H = trace/2*I + H_su2 decomposition consistency."""
        H = self.matrix(0.0)
        tr_half = 0.5 * np.trace(H)
        su2 = self.to_su2()
        H_su2 = np.array(
            [[su2.Omega(0.0), su2.omega(0.0)], [np.conj(su2.omega(0.0)), -su2.Omega(0.0)]],
            dtype=complex,
        )
        return bool(np.allclose(H, tr_half * np.eye(2, dtype=complex) + H_su2, atol=atol, rtol=0.0))


class FarhiGutmannHamiltonian(GQSHamiltonian):
    """Farhi-Gutmann special case: alpha=delta=1, beta=0."""

    def __init__(self, x: float, E: float, hbar: float = 1.0) -> None:
        super().__init__(alpha=1.0, beta=0.0 + 0.0j, delta=1.0, x=x, E=E, hbar=hbar)

    def t_star(self) -> float:
        return float(np.pi * self.hbar / (2.0 * self.E * self.x))

    def P_max(self) -> float:
        return 1.0

    def transition_probability(self, t: float) -> float:
        theta = self.E * self.x * t / self.hbar
        return float(self.x**2 * np.cos(theta) ** 2 + np.sin(theta) ** 2)

    def rabi_correspondence(self) -> dict[str, float]:
        x = self.x
        E = self.E
        return {
            "E1": float(E * (1.0 - x**2)),
            "E2": float(E * (1.0 + x**2)),
            "Gamma": float(E * x * np.sqrt(1.0 - x**2)),
            "omega_21": float(2.0 * E * x**2 / self.hbar),
            "omega_drive": 0.0,
        }


class RabiHamiltonian:
    """Driven two-level Rabi Hamiltonian (paper Section III)."""

    def __init__(
        self,
        E1: float,
        E2: float,
        Gamma: float,
        omega: float,
        x: float,
        hbar: float = 1.0,
    ) -> None:
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"x must be in [0, 1], got {x}")
        if E2 <= E1:
            raise ValueError(f"Require E2 > E1, got E1={E1}, E2={E2}")
        if hbar <= 0.0:
            raise ValueError(f"hbar must be positive, got {hbar}")

        self.E1 = float(E1)
        self.E2 = float(E2)
        self.Gamma = float(Gamma)
        self.omega = float(omega)
        self.x = float(x)
        self.hbar = float(hbar)
        self.omega_21 = (self.E2 - self.E1) / self.hbar

    @property
    def is_time_dependent(self) -> bool:
        return not np.isclose(self.omega, 0.0, atol=1e-15, rtol=0.0)

    @classmethod
    def from_magnetic_field(
        cls,
        B0: float,
        B1: float,
        omega: float,
        x: float | None = None,
        hbar: float = 1.0,
    ) -> "RabiHamiltonian":
        E_half = BOHR_MAGNETON * B0
        E1, E2 = -E_half, E_half
        Gamma = BOHR_MAGNETON * B1
        if x is None:
            x = overlap_from_fields(B0, B1)
        return cls(E1=E1, E2=E2, Gamma=Gamma, omega=omega, x=x, hbar=hbar)

    def matrix(self, t: float) -> np.ndarray:
        return np.array(
            [
                [self.E1, self.Gamma * np.exp(1j * self.omega * t)],
                [self.Gamma * np.exp(-1j * self.omega * t), self.E2],
            ],
            dtype=complex,
        )

    def interaction_matrix(self, t: float) -> np.ndarray:
        delta = self.omega - self.omega_21
        return np.array(
            [
                [0.0, self.Gamma * np.exp(1j * delta * t)],
                [self.Gamma * np.exp(-1j * delta * t), 0.0],
            ],
            dtype=complex,
        ) / self.hbar

    def energy_gap(self, t: float) -> float:
        vals = np.linalg.eigvalsh(self.matrix(t))
        return float(vals[1] - vals[0])

    def transition_probability(self, t: float) -> float:
        delta = self.omega - self.omega_21
        Omega = np.sqrt(self.Gamma**2 / self.hbar**2 + delta**2 / 4.0)
        D = (delta / 2.0 * self.x - self.Gamma / self.hbar * np.sqrt(1.0 - self.x**2)) / Omega
        return float(self.x**2 * np.cos(Omega * t) ** 2 + D**2 * np.sin(Omega * t) ** 2)

    def is_resonant(self, tol: float = 1e-10) -> bool:
        return abs(self.omega - self.omega_21) < tol

    def to_su2(self) -> SU2Fields:
        trace_half = 0.5 * (self.E1 + self.E2)
        Omega0 = 0.5 * (self.E1 - self.E2)

        _ = trace_half  # dropped global phase contribution
        return SU2Fields(
            omega=lambda t: self.Gamma * np.exp(1j * self.omega * t),
            Omega=lambda t: float(Omega0),
            T=None,
        )

    def verify_su2_mapping(self, t: float, atol: float = 1e-10) -> bool:
        """Check time-dependent H(t)=trace/2*I + H_su2(t) decomposition."""
        H = self.matrix(t)
        tr_half = 0.5 * np.trace(H)
        su2 = self.to_su2()
        H_su2 = np.array(
            [[su2.Omega(t), su2.omega(t)], [np.conj(su2.omega(t)), -su2.Omega(t)]],
            dtype=complex,
        )
        return bool(np.allclose(H, tr_half * np.eye(2, dtype=complex) + H_su2, atol=atol, rtol=0.0))


__all__ = [
    "FarhiGutmannHamiltonian",
    "GQSHamiltonian",
    "RabiHamiltonian",
]
