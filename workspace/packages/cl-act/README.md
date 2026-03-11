# cl-act

## Overview
`cl-act` is a staged OmegaX package wrapper for the `activation_cl_validation` Python library, which contains synthetic continual-learning validation tooling, analyses, plotting utilities, and symbolic checks used by the current deliverables.

Current release status is **blocked** because the latest review artifact (`phase_outputs/review_feedback.json`) reports decision `iterate`. Packaging metadata is prepared for reuse, but final release promotion is gated on an approved review decision.

## Installation
Canonical install target (once approved):

```bash
pip install activation-cl-validation
```

Current source-install command for local smoke testing:

```bash
pip install -e /app/outputs/runs/f1be1c20-5f6f-4b5c-85c2-3440e8b965c9/workspace/experiments/activation_cl_validation
```

## Configuration
Runtime dependencies are listed in `requirements.txt`.

Optional dependency:
- `pypdfium2` for PDF preview/raster checks used by visualization QA.

No environment variables are required by default.

## Usage Examples
Run experiment pipeline from source checkout:

```bash
PYTHONPATH=experiments/activation_cl_validation/src \
  experiments/.venv/bin/python experiments/activation_cl_validation/run_experiments.py \
  --output-dir experiments/activation_cl_validation/outputs \
  --figure-dir paper/figures \
  --table-dir paper/tables \
  --data-dir paper/data \
  --config experiments/activation_cl_validation/configs/simulation_config.json
```

Import API modules:

```python
from activation_cl_validation import core, analysis, plotting, sympy_checks
```

## Troubleshooting
- If import fails, verify the package is installed in the active interpreter and `python -c "import activation_cl_validation"` succeeds.
- If plotting fails, ensure `matplotlib`, `seaborn`, and `pandas` are installed from `requirements.txt`.
- If symbolic checks fail, verify `sympy` version compatibility (`>=1.12,<2`).
- If review gate remains `iterate`, treat this package as pre-release metadata only; do not publish externally.
