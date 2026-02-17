"""Tests for Classiq qDRIFT backend.

Classiq SDK tests are skipped when the package is not installed.
The Pauli conversion helper is tested independently (no classiq dependency).
"""

import numpy as np
import pytest

from hamover.embedding import encode_search, pauli_string_to_label


# ---------------------------------------------------------------------------
# Pauli conversion helper (no classiq dependency required)
# ---------------------------------------------------------------------------


def test_pauli_conversion_identity_term() -> None:
    """Identity Pauli string produces correct label."""
    label = pauli_string_to_label("I", n_qubits=2)
    assert label == ("I", "I")


def test_pauli_conversion_single_qubit() -> None:
    label = pauli_string_to_label("X0", n_qubits=2)
    assert label == ("X", "I")


def test_pauli_conversion_multi_qubit() -> None:
    label = pauli_string_to_label("X0*Z1", n_qubits=2)
    assert label == ("X", "Z")


def test_pauli_decomposition_roundtrip() -> None:
    """Pauli decomposition of an embedding Hamiltonian produces non-empty terms."""
    embedding = encode_search(N=2, target_index=0, penalty_strength=1.0)
    from hamover.core.types import SU2Fields

    fields = SU2Fields(
        omega=lambda _t: 0.5 + 0.0j,
        Omega=lambda _t: -0.3,
        T=1.0,
    )
    coeffs = embedding.pauli_coefficients_at(fields, t=0.5)
    assert len(coeffs) > 0
    # All coefficients should have negligible imaginary parts (Hermitian H)
    for _pstring, coeff in coeffs:
        assert abs(np.imag(coeff)) < 1e-8


# ---------------------------------------------------------------------------
# Full Classiq backend (skipped if classiq is not installed)
# ---------------------------------------------------------------------------

classiq = pytest.importorskip("classiq")

from hamover.backends.classiq_backend import (  # noqa: E402
    ClassiqResult,
    compile_classiq_circuit,
    run_classiq_search,
)


@pytest.mark.slow
def test_classiq_compile_n2() -> None:
    """Compile a circuit for N=2 Farhi-Gutmann — verify it synthesizes."""
    embedding = encode_search(N=2, target_index=0, penalty_strength=1.0)
    x = 1.0 / np.sqrt(2)

    qprog = compile_classiq_circuit(
        embedding=embedding,
        omega_func=lambda _t: 0.5 + 0.0j,
        Omega_func=lambda _t: -0.3,
        x=x,
        T=1.0,
        n_segments=5,
        num_qdrift=5,
    )
    assert qprog is not None


@pytest.mark.slow
def test_classiq_run_n2() -> None:
    """End-to-end run for N=2 with static fields."""
    embedding = encode_search(N=2, target_index=0, penalty_strength=1.0)
    x = 1.0 / np.sqrt(2)

    result = run_classiq_search(
        embedding=embedding,
        omega_func=lambda _t: 0.5 + 0.0j,
        Omega_func=lambda _t: -0.3,
        x=x,
        T=1.0,
        n_segments=5,
        num_qdrift=5,
        shots=100,
    )

    assert isinstance(result, ClassiqResult)
    assert 0.0 <= result.success_probability <= 1.0
    assert result.found_count + result.not_found_count + result.leakage_count == 100
    assert len(result.bitstrings) == 100
    assert result.n_qubits == 1
    assert result.simulation_method == "qdrift"
    assert result.method_params["num_qdrift"] == 5
