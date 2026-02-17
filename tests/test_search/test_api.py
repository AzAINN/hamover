import numpy as np
from types import SimpleNamespace

from hamover import HamoverSearch


def test_hamover_search_adiabatic_none() -> None:
    search = HamoverSearch(N=8, target=3)
    result = search.setup(approach="adiabatic", backend="none", epsilon=0.5).solve()

    assert 0.0 <= result.probability <= 1.0
    assert result.runtime > 0.0
    assert result.approach == "adiabatic"
    assert result.backend == "none"
    assert result.scaling_exponent == 0.5


def test_hamover_search_simulated_annealing_path() -> None:
    search = HamoverSearch(N=4, target=0, items=["a", "b", "c", "d"])
    result = search.setup(
        approach="adiabatic",
        backend="sim_annealing",
        epsilon=0.5,
        n_points=121,
        n_segments=24,
        shots=100,
        rng_seed=1,
    ).solve()

    assert 0.0 <= result.probability <= 1.0
    assert result.selected_index is None or (0 <= result.selected_index < 4)
    assert "validation_error_embedded" in result.diagnostics


def test_hamover_search_farhi_gutmann() -> None:
    search = HamoverSearch(N=16, target=2)
    result = search.setup(approach="farhi_gutmann", backend="none", E=1.0).solve()

    assert result.scaling_exponent == 0.5
    assert result.runtime > 0.0
    assert 0.0 <= result.probability <= 1.0


def test_hamover_search_nonadiabatic_none() -> None:
    search = HamoverSearch(N=8, target=1)
    result = search.setup(
        approach="nonadiabatic",
        backend="none",
        T=8.0,
        gamma=0.4,
        alpha=1.5,
    ).solve()

    assert 0.0 <= result.probability <= 1.0
    assert result.runtime == 8.0
    assert result.approach == "nonadiabatic"


def test_hamover_search_monotonic_none() -> None:
    search = HamoverSearch(N=8, target=1)
    result = search.setup(
        approach="monotonic",
        backend="none",
        scenario=2,
        omega_0=1.0,
    ).solve()

    assert 0.0 <= result.probability <= 1.0
    assert result.runtime > 0.0
    assert result.approach == "monotonic"


def test_hamover_search_custom_su2() -> None:
    omega = lambda _t: 0.25 + 0.0j
    Omega = lambda _t: -0.1

    search = HamoverSearch(N=8, target=1)
    result = search.setup(
        approach="custom_su2",
        backend="none",
        omega_func=omega,
        Omega_func=Omega,
        T=2.0,
    ).solve()

    assert np.isfinite(result.probability)
    assert result.runtime == 2.0


def test_classiq_default_penalty_is_zero(monkeypatch) -> None:
    captured: dict[str, float] = {}

    def _fake_run_classiq_search(*, embedding, **_kwargs):
        captured["penalty_strength"] = embedding.penalty_strength
        return SimpleNamespace(
            success_probability=0.6,
            found_count=60,
            not_found_count=40,
            leakage_count=0,
            bitstrings=[],
            circuit_depth=0,
            cx_count=0,
            simulation_method="qdrift",
            method_params={"num_qdrift": 20},
        )

    monkeypatch.setattr("hamover.search.run_classiq_search", _fake_run_classiq_search)

    result = HamoverSearch(N=8, target=1).setup(
        approach="adiabatic",
        backend="classiq",
        epsilon=0.5,
        n_points=121,
        n_segments=20,
        shots=100,
    ).solve()

    assert result.probability == 0.6
    assert captured["penalty_strength"] == 0.0


def test_classiq_penalty_override_respected(monkeypatch) -> None:
    captured: dict[str, float] = {}

    def _fake_run_classiq_search(*, embedding, **_kwargs):
        captured["penalty_strength"] = embedding.penalty_strength
        return SimpleNamespace(
            success_probability=0.6,
            found_count=60,
            not_found_count=40,
            leakage_count=0,
            bitstrings=[],
            circuit_depth=0,
            cx_count=0,
            simulation_method="qdrift",
            method_params={"num_qdrift": 20},
        )

    monkeypatch.setattr("hamover.search.run_classiq_search", _fake_run_classiq_search)

    HamoverSearch(N=8, target=1).setup(
        approach="adiabatic",
        backend="classiq",
        epsilon=0.5,
        n_points=121,
        n_segments=20,
        shots=100,
        penalty_strength=1.75,
    ).solve()

    assert captured["penalty_strength"] == 1.75
