"""Preset nonadiabatic control profiles for f(t), g(t) pathway experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class NonadiabaticPair:
    """Container for independent nonadiabatic controls f(t), g(t)."""

    f: Callable[[float], float]
    g: Callable[[float], float]
    T: float

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")


@dataclass(slots=True)
class OscillatoryPair:
    """Oscillatory nonadiabatic control pair (periodic behavior)."""

    T: float
    omega: float | None = None
    amplitude_f: float = 1.0
    amplitude_g: float = 1.0

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.omega is None:
            self.omega = 2.0 * np.pi / self.T

    def f(self, t: float) -> float:
        return float(self.amplitude_f * np.cos(self.omega * t))

    def g(self, t: float) -> float:
        return float(self.amplitude_g * np.sin(self.omega * t))

    def pair(self) -> NonadiabaticPair:
        return NonadiabaticPair(f=self.f, g=self.g, T=self.T)


@dataclass(slots=True)
class ConvergentPair:
    """Convergent nonadiabatic pair with decaying oscillations."""

    T: float
    alpha: float
    gamma: float

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")

    def f(self, t: float) -> float:
        return float(np.exp(-self.gamma * t) * np.cos(self.alpha * t))

    def g(self, t: float) -> float:
        return float(1.0 - np.exp(-self.gamma * t))

    def pair(self) -> NonadiabaticPair:
        return NonadiabaticPair(f=self.f, g=self.g, T=self.T)


__all__ = ["ConvergentPair", "NonadiabaticPair", "OscillatoryPair"]
