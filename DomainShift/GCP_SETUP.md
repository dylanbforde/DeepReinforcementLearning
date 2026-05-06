# GCP Experiment Runs

This checkout now runs all experiment variants from `main`; branch checkouts are no longer required.

## Setup

Install the project dependencies, then run the desired experiment from the repository root:

```bash
uv sync
```

Use `--cloud` to disable interactive rendering.

## Experiments

```bash
python DomainShift/Main.py --env bipedal --mode dsp --cloud --trials 10 --episodes 1000 --output-dir results/bipedal_dsp
python DomainShift/Main.py --env bipedal --mode control --cloud --trials 10 --episodes 1000 --output-dir results/bipedal_control
python DomainShift/Main.py --env cartpole --mode dsp --cloud --trials 10 --episodes 500 --output-dir results/cartpole_dsp
python DomainShift/Main.py --env cartpole --mode control --cloud --trials 10 --episodes 500 --output-dir results/cartpole_control
```

Each run writes CSV logs, the Optuna study database, trained model, and local plots under its `--output-dir`.
