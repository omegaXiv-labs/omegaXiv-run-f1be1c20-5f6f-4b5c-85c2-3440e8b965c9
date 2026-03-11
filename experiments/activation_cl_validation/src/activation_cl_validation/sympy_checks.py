from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import sympy as sp


def run_sympy_checks(output_path: Path) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rho, v0, c1, c2, Delta, eta_f, t = sp.symbols("rho v0 c1 c2 Delta eta_f t", positive=True)
    fixed_point_expr = rho**t * v0 + (c1 * Delta + c2 * eta_f**2) * (1 - rho**t) / (1 - rho)

    # Enforce contraction assumption 0 < rho < 1 using rho = exp(-k), k > 0.
    k = sp.symbols("k", positive=True)
    fixed_sub = sp.simplify(fixed_point_expr.subs(rho, sp.exp(-k)))
    steady_state_limit = sp.simplify(sp.limit(fixed_sub, t, sp.oo))
    steady_state_expected = sp.simplify((c1 * Delta + c2 * eta_f**2) / (1 - sp.exp(-k)))

    c, gamma, T = sp.symbols("c gamma T", positive=True)
    implication = sp.simplify((c * T * gamma) / T)

    checks: List[Dict[str, object]] = [
        {
            "check_id": "h1_fixed_point_limit_under_contraction",
            "expression": str(fixed_sub),
            "steady_state_limit": str(steady_state_limit),
            "expected_form": str(steady_state_expected),
            "pass": bool(sp.simplify(steady_state_limit - steady_state_expected) == 0),
        },
        {
            "check_id": "h4_regret_to_forgetting_implication",
            "expression": "R_dyn >= c*T*gamma and F >= R_dyn/T implies F >= c*gamma",
            "simplified_rhs": str(implication),
            "expected_form": "c*gamma",
            "pass": str(implication) == "c*gamma",
        },
    ]

    report = {
        "spec_path": "phase_outputs/SYMPY.md",
        "checks": checks,
        "all_passed": all(bool(x["pass"]) for x in checks),
        "assumptions": ["k > 0", "rho = exp(-k)", "equivalently 0 < rho < 1"],
    }

    import json

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
