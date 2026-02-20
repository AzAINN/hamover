"""Classiq gate-based execution path: qDRIFT and Suzuki-Trotter.

This backend compiles the embedded Hamiltonian to a gate-based circuit using
Classiq's Hamiltonian simulation primitives, then executes and decodes.
Unlike the D-Wave/IonQ backends, these preserve *all* Pauli terms — no
term-dropping.

Supported methods:
  - qDRIFT: randomized product formula (stochastic, lower depth)
  - suzuki_trotter: deterministic product formula (higher accuracy, deeper)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from hamover.core.types import SU2Fields
from hamover.embedding import HamiltonianEmbedding


def _bit_reverse_index(idx: int, n_qubits: int) -> int:
    """Return index with binary digits reversed (little ↔ big endian)."""
    return int(format(idx, f"0{n_qubits}b")[::-1], 2)


def _bit_reverse_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Reorder statevector amplitudes to match reversed bit significance."""
    n = int(np.log2(amplitudes.size))
    out = np.empty_like(amplitudes)
    for i in range(amplitudes.size):
        out[_bit_reverse_index(i, n)] = amplitudes[i]
    return out


@dataclass(slots=True)
class ClassiqResult:
    """Decoded measurement summary from Classiq execution."""

    success_probability: float
    found_count: int
    not_found_count: int
    leakage_count: int
    bitstrings: list[str]
    circuit_depth: int
    cx_count: int
    n_qubits: int
    simulation_method: str
    method_params: dict = field(default_factory=dict)


def _hamover_pauli_to_classiq(
    pauli_terms: list[tuple[str, complex]],
    n_qubits: int,
    imag_tol: float = 1e-8,
):
    """Convert hamover Pauli decomposition to Classiq SparsePauliOp.

    Hamover format: [("X0*Z2", coeff), ("I", coeff), ...]
    Classiq format: SparsePauliOp with SparsePauliTerm list.
    """
    from classiq import Pauli
    from classiq.qmod.builtins.structs import IndexedPauli, SparsePauliOp, SparsePauliTerm

    pauli_map = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}

    sparse_terms: list[SparsePauliTerm] = []
    for pstring, coeff in pauli_terms:
        c = float(np.real(coeff))
        if abs(np.imag(coeff)) > imag_tol:
            import warnings

            warnings.warn(
                f"Dropping imaginary part {np.imag(coeff):.2e} of Pauli coeff "
                f"for term '{pstring}'",
                RuntimeWarning,
                stacklevel=2,
            )
        if abs(c) < 1e-15:
            continue

        indexed: list[IndexedPauli] = []
        if pstring == "I":
            pass
        else:
            for token in pstring.split("*"):
                axis = token[0]
                qidx = int(token[1:])
                # Classiq treats index 0 as least significant; hamover uses MSB-first.
                mapped_idx = n_qubits - 1 - qidx
                indexed.append(IndexedPauli(pauli=pauli_map[axis], index=mapped_idx))

        sparse_terms.append(SparsePauliTerm(paulis=indexed, coefficient=c))

    return SparsePauliOp(terms=sparse_terms, num_qubits=n_qubits)


def _is_time_independent(
    embedding: HamiltonianEmbedding,
    fields: SU2Fields,
    T: float,
) -> bool:
    """Check if the embedded Hamiltonian is time-independent."""
    coeffs_0 = embedding.pauli_coefficients_at(fields, 0.0)
    coeffs_mid = embedding.pauli_coefficients_at(fields, T / 2.0)
    if len(coeffs_0) != len(coeffs_mid):
        return False
    return all(
        np.isclose(c1, c2, atol=1e-10)
        for (_, c1), (_, c2) in zip(coeffs_0, coeffs_mid)
    )


def compile_classiq_circuit(
    embedding: HamiltonianEmbedding,
    omega_func: Callable[[float], complex],
    Omega_func: Callable[[float], float],
    x: float,
    T: float,
    n_segments: int = 100,
    simulation_method: str = "qdrift",
    num_qdrift: int = 20,
    trotter_order: int = 2,
    trotter_reps: int = 10,
):
    """Build a Classiq quantum program for the embedded search Hamiltonian.

    Parameters
    ----------
    simulation_method : str
        "qdrift" or "suzuki_trotter"
    num_qdrift : int
        Number of qDRIFT random samples (only for qdrift method)
    trotter_order : int
        Suzuki-Trotter order (1, 2, or 4; only for suzuki_trotter method)
    trotter_reps : int
        Number of Trotter repetitions (only for suzuki_trotter method)

    Returns the synthesized quantum program ready for execution.
    """
    from classiq import (
        Output,
        QArray,
        QBit,
        create_model,
        prepare_amplitudes,
        qfunc,
        synthesize,
    )

    if simulation_method not in {"qdrift", "suzuki_trotter"}:
        raise ValueError(
            f"Unknown simulation_method '{simulation_method}'. "
            "Use 'qdrift' or 'suzuki_trotter'."
        )

    fields = SU2Fields(omega=omega_func, Omega=Omega_func, T=T)
    n_qubits = embedding.n_qubits

    # Compute initial state amplitudes (must be real for prepare_amplitudes)
    psi0 = embedding.initial_state(x)
    # Classiq expects amplitudes in little-endian order; embedding uses big-endian.
    psi0_reordered = _bit_reverse_amplitudes(psi0)
    amplitudes = np.real(psi0_reordered).tolist()

    time_independent = _is_time_independent(embedding, fields, T)

    if time_independent:
        coeffs_0 = embedding.pauli_coefficients_at(fields, 0.0)
        pauli_op = _hamover_pauli_to_classiq(coeffs_0, n_qubits)

        if simulation_method == "qdrift":
            from classiq import qdrift as qdrift_fn

            @qfunc
            def main(q: Output[QArray[QBit]]) -> None:
                prepare_amplitudes(amplitudes, 0.0, q)
                qdrift_fn(pauli_op, evolution_coefficient=T,
                          num_qdrift=num_qdrift, qbv=q)
        else:
            from classiq import suzuki_trotter as st_fn

            @qfunc
            def main(q: Output[QArray[QBit]]) -> None:
                prepare_amplitudes(amplitudes, 0.0, q)
                st_fn(pauli_op, evolution_coefficient=T,
                      order=trotter_order, repetitions=trotter_reps, qbv=q)
    else:
        dt = T / n_segments
        segment_ops = []
        for k in range(n_segments):
            t_k = (k + 0.5) * dt
            coeffs_k = embedding.pauli_coefficients_at(fields, t_k)
            op_k = _hamover_pauli_to_classiq(coeffs_k, n_qubits)
            segment_ops.append(op_k)

        if simulation_method == "qdrift":
            from classiq import qdrift as qdrift_fn

            @qfunc
            def main(q: Output[QArray[QBit]]) -> None:
                prepare_amplitudes(amplitudes, 0.0, q)
                for op_k in segment_ops:
                    qdrift_fn(op_k, evolution_coefficient=dt,
                              num_qdrift=num_qdrift, qbv=q)
        else:
            from classiq import suzuki_trotter as st_fn

            @qfunc
            def main(q: Output[QArray[QBit]]) -> None:
                prepare_amplitudes(amplitudes, 0.0, q)
                for op_k in segment_ops:
                    st_fn(op_k, evolution_coefficient=dt,
                          order=trotter_order, repetitions=trotter_reps, qbv=q)

    model = create_model(main)
    quantum_program = synthesize(model)
    return quantum_program


def run_classiq_search(
    embedding: HamiltonianEmbedding,
    omega_func: Callable[[float], complex],
    Omega_func: Callable[[float], float],
    x: float,
    T: float,
    n_segments: int = 100,
    simulation_method: str = "qdrift",
    num_qdrift: int = 20,
    trotter_order: int = 2,
    trotter_reps: int = 10,
    shots: int = 1000,
) -> ClassiqResult:
    """Compile, execute on Classiq, and decode measurement outcomes.

    Parameters
    ----------
    simulation_method : str
        "qdrift" or "suzuki_trotter"
    num_qdrift : int
        Number of qDRIFT samples (qdrift only)
    trotter_order : int
        Suzuki-Trotter order (suzuki_trotter only)
    trotter_reps : int
        Number of Trotter repetitions (suzuki_trotter only)
    """
    from classiq import execute, set_quantum_program_execution_preferences
    from classiq.execution import ExecutionPreferences

    quantum_program = compile_classiq_circuit(
        embedding=embedding,
        omega_func=omega_func,
        Omega_func=Omega_func,
        x=x,
        T=T,
        n_segments=n_segments,
        simulation_method=simulation_method,
        num_qdrift=num_qdrift,
        trotter_order=trotter_order,
        trotter_reps=trotter_reps,
    )

    preferences = ExecutionPreferences(num_shots=shots)
    quantum_program = set_quantum_program_execution_preferences(quantum_program, preferences)
    job = execute(quantum_program)
    results = job.result()
    parsed_counts = results[0].value.parsed_counts

    found = 0
    not_found = 0
    leakage = 0
    all_bitstrings: list[str] = []

    for parsed_state in parsed_counts:
        qubit_vals = parsed_state.state["q"]
        count = parsed_state.shots
        # Reverse to restore MSB-first convention used by embedding.decode()
        bitstring = "".join(str(int(b)) for b in reversed(qubit_vals)).zfill(embedding.n_qubits)
        for _ in range(count):
            all_bitstrings.append(bitstring)
            tag = embedding.decode(bitstring)
            if tag == "w":
                found += 1
            elif tag == "r":
                not_found += 1
            else:
                leakage += 1

    denom = found + not_found
    p_succ = float(found / denom) if denom > 0 else 0.0

    circuit_depth = 0
    cx_count = 0
    try:
        transpiled = quantum_program.transpiled_circuit
        circuit_depth = transpiled.depth if hasattr(transpiled, "depth") else 0
        cx_count = transpiled.count_ops.get("cx", 0) if hasattr(transpiled, "count_ops") else 0
    except Exception:
        pass

    method_params = {}
    if simulation_method == "qdrift":
        method_params = {"num_qdrift": num_qdrift}
    else:
        method_params = {"trotter_order": trotter_order, "trotter_reps": trotter_reps}

    return ClassiqResult(
        success_probability=p_succ,
        found_count=found,
        not_found_count=not_found,
        leakage_count=leakage,
        bitstrings=all_bitstrings,
        circuit_depth=circuit_depth,
        cx_count=cx_count,
        n_qubits=embedding.n_qubits,
        simulation_method=simulation_method,
        method_params=method_params,
    )
