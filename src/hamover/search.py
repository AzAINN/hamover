"""Public entry point for continuous-time unstructured search with hamover."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Optional

import numpy as np

from hamover.embedding import auto_penalty, encode_search
from hamover.simuq_interface import build_simuq_system, validate_simuq_system
from hamover.backends import (
    compile_ionq_program,
    compile_quera_program,
    run_classiq_search,
    run_simulated_annealing,
    simulate_embedded,
)
from hamover.core.types import SearchProblem, SearchResult, SearchSetup
from hamover.hamiltonians.controlled import MonotonicScenario1, MonotonicScenario2, SU2Hamiltonian
from hamover.hamiltonians.scheduled import AdiabaticSearchHamiltonian, NonadiabaticSearchHamiltonian
from hamover.hamiltonians.static import FarhiGutmannHamiltonian
from hamover.schedules.library import ConstantSpeedSchedule, GapPowerSchedule, RolandCerfSchedule
from hamover.schedules.nonadiabatic import ConvergentPair
from hamover.solver import evolve, solve_monotonic


class HamoverSearch:
    """Single entry-point class that mirrors the QHDOPT-style workflow."""

    def __init__(self, N: int, target: int, items: Optional[list[Any]] = None) -> None:
        self.problem = SearchProblem(N=N, target_index=target, items=items)
        self._setup = SearchSetup()

    @classmethod
    def define(cls, N: int, target_index: int, items: Optional[list[Any]] = None) -> "HamoverSearch":
        """Alternative constructor used by the architecture docs."""
        return cls(N=N, target=target_index, items=items)

    def setup(self, approach: str = "adiabatic", backend: str = "none", **options: Any) -> "HamoverSearch":
        self._setup = SearchSetup(approach=approach, backend=backend, options=dict(options))
        return self

    def solve(self, T: float | None = None) -> SearchResult:
        x = float(1.0 / np.sqrt(self.problem.N))
        approach = self._setup.approach.lower().strip()
        backend = self._setup.backend.lower().strip()
        opts = dict(self._setup.options)
        if T is not None:
            opts["T"] = float(T)

        if approach == "monotonic":
            return self._solve_monotonic(x=x, backend=backend, **opts)
        return self._solve_general(x=x, approach=approach, backend=backend, **opts)

    def _solve_general(self, x: float, approach: str, backend: str, **opts: Any) -> SearchResult:
        n_points = int(opts.get("n_points", 401))
        n_segments = int(opts.get("n_segments", 100))
        runtime, ham, su2, scaling = self._build_model(approach=approach, x=x, **opts)

        sch = evolve(ham, x=x, T=runtime, n_points=n_points)
        p_ref = float(sch.probability_sw[-1])

        diagnostics: dict[str, Any] = {
            "reference_probability_final": p_ref,
            "norm_deviation_max": float(sch.norm_deviation_max),
            "simuq_qsystem_built": False,
        }

        selected_index: Optional[int] = None
        selected_item: Optional[Any] = None
        result_probability = p_ref
        target_found = p_ref >= 0.5

        if backend != "none":
            # For gate-based Classiq simulation, a large leakage penalty inflates
            # Hamiltonian norm and can severely degrade qDRIFT/Trotter accuracy.
            default_penalty = 0.0 if backend == "classiq" else auto_penalty(su2.to_su2(), runtime)
            penalty_strength = float(opts.get("penalty_strength", default_penalty))
            embedding = encode_search(
                N=self.problem.N,
                target_index=self.problem.target_index,
                penalty_strength=penalty_strength,
            )
            built = build_simuq_system(
                embedding=embedding,
                omega_func=su2.omega,
                Omega_func=su2.Omega,
                T=runtime,
                n_segments=n_segments,
            )
            val = validate_simuq_system(
                built=built,
                embedding=embedding,
                omega_func=su2.omega,
                Omega_func=su2.Omega,
                x=x,
                T=runtime,
                n_points=n_points,
            )
            diagnostics.update(
                {
                    "embedded_probability_final": float(val.p_embedded_final),
                    "validation_error_embedded": float(val.error_embedded),
                    "validation_error_simuq": (
                        None if val.error_simuq is None else float(val.error_simuq)
                    ),
                    "max_leakage": float(val.max_leakage),
                    "simuq_qsystem_built": built.qsystem is not None,
                }
            )

            if backend in {"sim_annealing", "dwave"}:
                sa = run_simulated_annealing(
                    embedding=embedding,
                    omega_func=su2.omega,
                    Omega_func=su2.Omega,
                    T=runtime,
                    shots=int(opts.get("shots", 1000)),
                    strict_ising=bool(opts.get("strict_ising", False)),
                    rng_seed=opts.get("rng_seed", 0),
                )
                result_probability = float(sa.success_probability)
                diagnostics.update(
                    {
                        "backend_found_count": int(sa.found_count),
                        "backend_not_found_count": int(sa.not_found_count),
                        "backend_leakage_count": int(sa.leakage_count),
                        "backend_dropped_terms": int(sa.ising_model.dropped_terms),
                        "backend_dropped_weight": float(sa.ising_model.dropped_weight),
                    }
                )
                if sa.bitstrings:
                    valid = [bs for bs in sa.bitstrings if embedding.decode(bs) != "leakage"]
                    if valid:
                        bitstring = Counter(valid).most_common(1)[0][0]
                        selected_index = int(bitstring, 2)
                        target_found = selected_index == self.problem.target_index
            elif backend == "qutip":
                sim = simulate_embedded(
                    embedding=embedding,
                    omega_func=su2.omega,
                    Omega_func=su2.Omega,
                    x=x,
                    T=runtime,
                    n_points=n_points,
                )
                result_probability = float(sim.probability_w[-1])
                target_found = result_probability >= 0.5
                diagnostics["backend_max_leakage"] = float(sim.max_leakage)
            elif backend == "ionq":
                prog = compile_ionq_program(
                    embedding=embedding,
                    omega_func=su2.omega,
                    Omega_func=su2.Omega,
                    T=runtime,
                    n_segments=n_segments,
                )
                diagnostics["ionq_segments"] = len(prog.segments)
            elif backend == "quera":
                prog = compile_quera_program(
                    embedding=embedding,
                    omega_func=su2.omega,
                    Omega_func=su2.Omega,
                    T=runtime,
                    n_points=max(2, n_segments),
                )
                diagnostics["quera_points"] = len(prog.t)
            elif backend == "classiq":
                sim_method = str(opts.get("hamiltonian_simulation", "qdrift")).lower().strip()
                cq = run_classiq_search(
                    embedding=embedding,
                    omega_func=su2.omega,
                    Omega_func=su2.Omega,
                    x=x,
                    T=runtime,
                    n_segments=n_segments,
                    simulation_method=sim_method,
                    num_qdrift=int(opts.get("num_qdrift", 20)),
                    trotter_order=int(opts.get("trotter_order", 2)),
                    trotter_reps=int(opts.get("trotter_reps", 10)),
                    shots=int(opts.get("shots", 1000)),
                )
                result_probability = float(cq.success_probability)
                classiq_diag = {
                    "backend_found_count": int(cq.found_count),
                    "backend_not_found_count": int(cq.not_found_count),
                    "backend_leakage_count": int(cq.leakage_count),
                    "circuit_depth": int(cq.circuit_depth),
                    "cx_count": int(cq.cx_count),
                    "simulation_method": cq.simulation_method,
                    "method_params": cq.method_params,
                }
                # Backward-compatible flattened fields used in notebook diagnostics.
                if cq.simulation_method == "qdrift" and "num_qdrift" in cq.method_params:
                    classiq_diag["num_qdrift"] = int(cq.method_params["num_qdrift"])
                if cq.simulation_method == "suzuki_trotter":
                    if "trotter_order" in cq.method_params:
                        classiq_diag["trotter_order"] = int(cq.method_params["trotter_order"])
                    if "trotter_reps" in cq.method_params:
                        classiq_diag["trotter_reps"] = int(cq.method_params["trotter_reps"])
                diagnostics.update(classiq_diag)
                if cq.bitstrings:
                    valid = [bs for bs in cq.bitstrings if embedding.decode(bs) != "leakage"]
                    if valid:
                        bitstring = Counter(valid).most_common(1)[0][0]
                        selected_index = int(bitstring, 2)
                        target_found = selected_index == self.problem.target_index
            else:
                raise ValueError(
                    f"Unknown backend '{backend}'. Use one of: none, qutip, sim_annealing, dwave, ionq, quera, classiq."
                )

        if selected_index is None and result_probability >= 0.5:
            selected_index = self.problem.target_index
        if selected_index is not None and self.problem.items is not None:
            selected_item = self.problem.items[selected_index]

        return SearchResult(
            target_found=bool(target_found),
            probability=float(result_probability),
            runtime=float(runtime),
            scaling_exponent=scaling,
            target_index=self.problem.target_index,
            selected_index=selected_index,
            selected_item=selected_item,
            overlap_x=x,
            approach=approach,
            backend=backend,
            diagnostics=diagnostics,
        )

    def _solve_monotonic(self, x: float, backend: str, **opts: Any) -> SearchResult:
        if backend != "none":
            raise ValueError("monotonic approach currently supports backend='none' only")

        scenario = int(opts.get("scenario", 1))
        omega_0 = float(opts.get("omega_0", 1.0))
        n_points = int(opts.get("n_points", 401))

        if scenario == 1:
            c = float(opts.get("c", x / np.sqrt(1.0 - x**2)))
            xi_opt = opts.get("xi")
            model = (
                MonotonicScenario1.monotonic(omega_0=omega_0, c=c)
                if xi_opt is None
                else MonotonicScenario1(omega_0=omega_0, xi=float(xi_opt), c=c)
            )
            runtime = float(opts.get("T", 5.0 / model.xi))
        elif scenario == 2:
            xi_opt = opts.get("xi")
            model = (
                MonotonicScenario2.monotonic(omega_0=omega_0)
                if xi_opt is None
                else MonotonicScenario2(omega_0=omega_0, xi=float(xi_opt))
            )
            runtime = float(opts.get("T", 15.0 / model.xi))
        else:
            raise ValueError("scenario must be 1 or 2")

        t = np.linspace(0.0, runtime, n_points)
        closed = solve_monotonic(model, t_array=t, x=x)
        p = float(closed.probability_sw[-1])

        diagnostics = {
            "monotonic_scenario": scenario,
            "reference_probability_final": p,
        }
        return SearchResult(
            target_found=p >= 0.5,
            probability=p,
            runtime=runtime,
            scaling_exponent=None,
            target_index=self.problem.target_index,
            selected_index=self.problem.target_index if p >= 0.5 else None,
            selected_item=(
                None
                if self.problem.items is None or p < 0.5
                else self.problem.items[self.problem.target_index]
            ),
            overlap_x=x,
            approach="monotonic",
            backend="none",
            diagnostics=diagnostics,
        )

    def _build_model(
        self,
        approach: str,
        x: float,
        **opts: Any,
    ) -> tuple[float, Any, SU2Hamiltonian, Optional[float]]:
        def _runtime(default: float) -> float:
            t_opt = opts.get("T")
            return float(default if t_opt is None else t_opt)

        if approach == "farhi_gutmann":
            E = float(opts.get("E", 1.0))
            ham = FarhiGutmannHamiltonian(x=x, E=E)
            runtime = _runtime(ham.t_star())
            su2_fields = ham.to_su2()
            su2 = SU2Hamiltonian(omega_func=su2_fields.omega, Omega_func=su2_fields.Omega, T=runtime)
            return runtime, ham, su2, 0.5

        if approach == "adiabatic":
            schedule_name = str(opts.get("schedule", "roland_cerf")).lower().strip()
            if schedule_name == "roland_cerf":
                schedule = RolandCerfSchedule(x=x, epsilon=float(opts.get("epsilon", 0.5)))
                scaling = 0.5
            elif schedule_name == "constant":
                if "T" not in opts:
                    raise ValueError("constant schedule requires T")
                schedule = ConstantSpeedSchedule(T=float(opts["T"]))
                scaling = None
            elif schedule_name == "gap_power":
                schedule = GapPowerSchedule(
                    x=x,
                    epsilon=float(opts.get("epsilon", 0.5)),
                    p=float(opts.get("p", 2.0)),
                )
                scaling = 0.5 if np.isclose(float(opts.get("p", 2.0)), 2.0) else None
            else:
                raise ValueError("schedule must be one of: roland_cerf, constant, gap_power")

            runtime = _runtime(schedule.T)
            ham = AdiabaticSearchHamiltonian(schedule=schedule, x=x)
            su2 = SU2Hamiltonian.from_adiabatic(schedule=schedule, x=x)
            return runtime, ham, su2, scaling

        if approach == "nonadiabatic":
            runtime = _runtime(10.0)
            if "f_func" in opts and "g_func" in opts:
                f_func = opts["f_func"]
                g_func = opts["g_func"]
            else:
                pair = ConvergentPair(
                    T=runtime,
                    alpha=float(opts.get("alpha", 2.0 * np.pi / runtime)),
                    gamma=float(opts.get("gamma", 1.0 / runtime)),
                ).pair()
                f_func, g_func = pair.f, pair.g
            ham = NonadiabaticSearchHamiltonian(f_func=f_func, g_func=g_func, x=x)
            su2 = SU2Hamiltonian.from_nonadiabatic(f_func=f_func, g_func=g_func, x=x, T=runtime)
            return runtime, ham, su2, None

        if approach == "custom_su2":
            if "omega_func" not in opts or "Omega_func" not in opts:
                raise ValueError("custom_su2 requires omega_func and Omega_func")
            if "T" not in opts:
                raise ValueError("custom_su2 requires T")
            runtime = float(opts["T"])
            su2 = SU2Hamiltonian(
                omega_func=opts["omega_func"],
                Omega_func=opts["Omega_func"],
                T=runtime,
            )
            return runtime, su2, su2, None

        raise ValueError(
            f"Unknown approach '{approach}'. Use one of: farhi_gutmann, adiabatic, nonadiabatic, monotonic, custom_su2."
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the configured problem/setup in plain dictionary form."""
        return {
            "problem": asdict(self.problem),
            "setup": asdict(self._setup),
        }


__all__ = ["HamoverSearch"]
