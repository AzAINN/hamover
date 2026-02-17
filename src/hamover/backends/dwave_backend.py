"""D-Wave-style Ising execution path with simulated annealing fallback.

This backend maps the embedded Hamiltonian to Ising terms (Z and ZZ), then
executes classical simulated annealing to emulate the final anneal readout.
It is intentionally aligned with the QHDOPT simulated-annealing workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from hamover.core.types import SU2Fields

from hamover.embedding import HamiltonianEmbedding, pauli_string_to_label


@dataclass(slots=True)
class IsingModel:
    """Ising representation H = offset + sum h_i z_i + sum J_ij z_i z_j."""

    h: dict[int, float]
    J: dict[tuple[int, int], float]
    offset: float
    dropped_terms: int
    dropped_weight: float


@dataclass(slots=True)
class AnnealScheduleTrace:
    """Derived anneal-shape diagnostics from su(2) control fields."""

    t: np.ndarray
    A: np.ndarray
    B: np.ndarray
    s: np.ndarray


@dataclass(slots=True)
class DWaveAnnealResult:
    """Decoded measurement summary for the simulated annealing pathway."""

    success_probability: float
    found_count: int
    not_found_count: int
    leakage_count: int
    bitstrings: list[str]
    ising_model: IsingModel
    schedule: AnnealScheduleTrace


def _is_z_only(label: tuple[str, ...]) -> bool:
    return all(axis in ("I", "Z") for axis in label)


def _z_support(label: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(i for i, axis in enumerate(label) if axis == "Z")


def extract_ising_model(
    embedding: HamiltonianEmbedding,
    su2_fields: SU2Fields,
    t: float,
    tol: float = 1e-10,
) -> IsingModel:
    """Extract Ising coefficients from the Pauli decomposition at a fixed time."""
    coeffs = embedding.pauli_coefficients_at(su2_fields, t, tol=tol)

    h: dict[int, float] = {}
    J: dict[tuple[int, int], float] = {}
    offset = 0.0
    dropped_terms = 0
    dropped_weight = 0.0

    for pstring, coeff in coeffs:
        c = float(np.real(coeff))
        label = pauli_string_to_label(pstring, embedding.n_qubits)

        if not _is_z_only(label):
            dropped_terms += 1
            dropped_weight += abs(c)
            continue

        support = _z_support(label)
        if len(support) == 0:
            offset += c
        elif len(support) == 1:
            i = support[0]
            h[i] = h.get(i, 0.0) + c
        elif len(support) == 2:
            i, j = support
            key = (i, j) if i < j else (j, i)
            J[key] = J.get(key, 0.0) + c
        else:
            # Higher-order Z terms are dropped in this first bridge version.
            dropped_terms += 1
            dropped_weight += abs(c)

    return IsingModel(
        h=h,
        J=J,
        offset=float(offset),
        dropped_terms=dropped_terms,
        dropped_weight=float(dropped_weight),
    )


def anneal_schedule_from_controls(
    omega_func,
    Omega_func,
    T: float,
    n_points: int = 200,
) -> AnnealScheduleTrace:
    """Convert su(2) controls to D-Wave-like A(s), B(s), s diagnostics."""
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    t = np.linspace(0.0, T, n_points)
    A_raw = np.array([abs(complex(omega_func(float(tt)))) for tt in t], dtype=float)
    B_raw = np.array([abs(float(Omega_func(float(tt)))) for tt in t], dtype=float)

    A_scale = float(np.max(A_raw)) if np.max(A_raw) > 0.0 else 1.0
    B_scale = float(np.max(B_raw)) if np.max(B_raw) > 0.0 else 1.0
    A = A_raw / A_scale
    B = B_raw / B_scale
    s = np.divide(B, A + B + 1e-12)

    return AnnealScheduleTrace(t=t, A=A, B=B, s=s)


def _spin_sample_to_bitstring(sample: dict[int, int], n_qubits: int) -> str:
    # Pauli-Z eigenvalue mapping: z=+1 -> |0>, z=-1 -> |1>
    bits = ["0"] * n_qubits
    for i in range(n_qubits):
        bits[i] = "0" if int(sample.get(i, 1)) > 0 else "1"
    return "".join(bits)


def _fallback_boltzmann_samples(
    model: IsingModel,
    n_qubits: int,
    shots: int,
    beta: float = 3.0,
    rng_seed: int | None = None,
) -> list[str]:
    rng = np.random.default_rng(rng_seed)
    configs = []
    energies = []
    for idx in range(1 << n_qubits):
        bitstring = format(idx, f"0{n_qubits}b")
        z = np.array([1.0 if b == "0" else -1.0 for b in bitstring], dtype=float)
        e = model.offset
        for i, hi in model.h.items():
            e += hi * z[i]
        for (i, j), Jij in model.J.items():
            e += Jij * z[i] * z[j]
        configs.append(bitstring)
        energies.append(e)

    energies_arr = np.asarray(energies, dtype=float)
    weights = np.exp(-beta * (energies_arr - np.min(energies_arr)))
    probs = weights / np.sum(weights)
    picks = rng.choice(len(configs), size=shots, p=probs)
    return [configs[i] for i in picks]


def run_simulated_annealing(
    embedding: HamiltonianEmbedding,
    omega_func,
    Omega_func,
    T: float,
    shots: int = 1000,
    t_sample: float | None = None,
    strict_ising: bool = False,
    rng_seed: int | None = None,
    **sampler_kwargs,
) -> DWaveAnnealResult:
    """Execute the simulated annealing pathway and decode outcomes."""
    if shots < 1:
        raise ValueError("shots must be >= 1")

    fields = SU2Fields(omega=omega_func, Omega=Omega_func, T=T)
    t_eval = float(T if t_sample is None else t_sample)
    model = extract_ising_model(embedding, fields, t_eval)

    if model.dropped_terms > 0:
        msg = (
            "Non-Ising Pauli terms were dropped during D-Wave mapping "
            f"(count={model.dropped_terms}, total_weight={model.dropped_weight:.3e})"
        )
        if strict_ising:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    bitstrings: list[str]
    try:
        from dwave.samplers import SimulatedAnnealingSampler

        sampler = SimulatedAnnealingSampler()
        response = sampler.sample_ising(model.h, model.J, num_reads=shots, **sampler_kwargs)
        bitstrings = []
        for sample in response.samples():
            bitstrings.append(_spin_sample_to_bitstring(sample, embedding.n_qubits))
    except Exception:
        bitstrings = _fallback_boltzmann_samples(
            model=model,
            n_qubits=embedding.n_qubits,
            shots=shots,
            rng_seed=rng_seed,
        )

    found = 0
    not_found = 0
    leakage = 0
    for bs in bitstrings:
        tag = embedding.decode(bs)
        if tag == "w":
            found += 1
        elif tag == "r":
            not_found += 1
        else:
            leakage += 1

    denom = found + not_found
    p_succ = float(found / denom) if denom > 0 else 0.0

    schedule = anneal_schedule_from_controls(omega_func, Omega_func, T)
    return DWaveAnnealResult(
        success_probability=p_succ,
        found_count=found,
        not_found_count=not_found,
        leakage_count=leakage,
        bitstrings=bitstrings,
        ising_model=model,
        schedule=schedule,
    )
