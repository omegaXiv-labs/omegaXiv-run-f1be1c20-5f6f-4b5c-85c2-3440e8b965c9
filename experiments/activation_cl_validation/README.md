# Activation CL Validation Simulation

This experiment package runs a reproducible synthetic validation of the theory-first hybrid plan:
- `h1` bounded-drift moment stability checks
- `h4` static-activation impossibility boundary checks
- Cross-hypothesis ablation bridge

All quantitative outputs in this package come from a synthetic simulator with seeded noise and symbolic checks. They are not produced by training models on the named benchmarks directly.

## Layout
- `src/activation_cl_validation/core.py`: synthetic stream generation and metric simulation
- `src/activation_cl_validation/analysis.py`: CI tables and acceptance snapshots
- `src/activation_cl_validation/plotting.py`: seaborn-styled multi-panel PDF figures + PDF raster checks
- `src/activation_cl_validation/sympy_checks.py`: symbolic theorem-template checks from `phase_outputs/SYMPY.md`
- `run_experiments.py`: thin CLI entrypoint
- `configs/simulation_config.json`: method and seed configuration
- `tests/test_core.py`: minimal reproducibility and invariant tests

## Setup
Create a virtual environment under `experiments/.venv`, then install the package and its declared dependencies:

```bash
python -m pip install -e experiments/activation_cl_validation
```

## Run
From workspace root:

```bash
PYTHONPATH=experiments/activation_cl_validation/src experiments/.venv/bin/python experiments/activation_cl_validation/run_experiments.py \
  --output-dir experiments/activation_cl_validation/outputs \
  --figure-dir paper/figures \
  --table-dir paper/tables \
  --data-dir paper/data \
  --config experiments/activation_cl_validation/configs/simulation_config.json
```

## Outputs
- `paper/figures/*.pdf`
- `paper/tables/*.csv`
- `paper/data/simulated_validation_runs.csv`
- `experiments/activation_cl_validation/outputs/results_summary.json`
- `experiments/activation_cl_validation/outputs/sympy_validation_report.json`
- `experiments/experiment_log.jsonl`
