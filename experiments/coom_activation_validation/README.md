# COOM Activation Validation

This experiment validates the paper's activation idea in a real COOM RL setting instead of the synthetic simulator used elsewhere in the repository.

## What it runs

- COOM single-task benchmarks with PPO
- `GELU` baseline versus an operationalized dual-timescale activation
- Periodic evaluation with reward and normalized COOM success
- Aggregate learning-curve plots
- A short demo video from the trained proposed-activation agent

## Environment

The dedicated environment is the repository root `.venv`.

## Typical command

```bash
./.venv/bin/python experiments/coom_activation_validation/run_benchmark.py \
  --benchmarks pitfall-default hide_and_seek-default \
  --seeds 0 1 \
  --timesteps 20000 \
  --eval-every 2500 \
  --eval-episodes 5 \
  --output-root artifacts/coom_activation_validation
```

## Notes

- `external/COOM` is expected to exist and is intentionally gitignored.
- The proposed activation is an executable operationalization of the paper's novelty/utility/slow-anchor idea for PPO. It is not a verbatim implementation of a previously released reference codebase, because the current repo only contains a synthetic validation package and no real RL implementation.
- Outputs are written under `artifacts/coom_activation_validation/` and are intentionally gitignored.

