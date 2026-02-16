"""Physical constants and conversion helpers used by hamover."""

# Reduced Planck constants
HBAR_EV_S = 6.582119514e-16  # eV*s
HBAR_J_S = 1.054571817e-34   # J*s
HBAR_NATURAL = 1.0           # paper convention

# Bohr magneton
BOHR_MAGNETON = 5.7883818012e-5  # eV/T

# Electron properties
ELECTRON_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31   # kg
SPEED_OF_LIGHT = 2.99792458e8      # m/s


def to_natural_units(E_eV: float, hbar: float = HBAR_EV_S) -> float:
    """Convert an energy scale (eV) to angular frequency (1/s) in hbar=1 units."""
    return E_eV / hbar


def from_natural_units(omega: float, hbar: float = HBAR_EV_S) -> float:
    """Convert an angular frequency (1/s) back to energy (eV)."""
    return omega * hbar


def overlap_from_fields(B0: float, B1: float) -> float:
    """Eq. 39 mapping x(B0, B1) = (B0/B1)/sqrt(1 + (B0/B1)^2)."""
    if B1 == 0:
        raise ValueError("B1 must be nonzero")
    ratio = B0 / B1
    return ratio / (1.0 + ratio**2) ** 0.5


def rabi_frequency(B0: float, B1: float | None = None) -> float | tuple[float, float]:
    """Return omega_21 and optional Gamma from magnetic fields using mu_B."""
    omega_21 = 2.0 * BOHR_MAGNETON * B0 / HBAR_EV_S
    if B1 is None:
        return omega_21

    Gamma = BOHR_MAGNETON * B1
    return omega_21, Gamma
