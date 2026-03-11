from __future__ import annotations

from activation_cl_validation.core import generate_scenarios, simulate_runs, theorem_assumption_checks


def test_simulation_generates_rows() -> None:
    scenarios = generate_scenarios()
    df = simulate_runs(methods=["Proposed full model", "GELU"], seeds=[7, 13], scenarios=scenarios[:3])
    assert not df.empty
    assert {"acc", "forgetting_index", "regret_slope"}.issubset(df.columns)


def test_assumption_checks_have_expected_keys() -> None:
    scenarios = generate_scenarios()
    df = simulate_runs(methods=["Proposed full model", "GELU", "Static GELU", "SELU", "ReLU"], seeds=[7, 13], scenarios=scenarios)
    checks = theorem_assumption_checks(df)
    assert {"check_id", "pass", "value"}.issubset(checks.columns)
    assert len(checks) >= 4
