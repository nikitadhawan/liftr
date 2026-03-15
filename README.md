# LIFTR: LInearized Function TRick

This codebase implements the experiments and reproduces the results of *Approximating Function Space Distance for Continual Learning in Transformers*.

LIFTR approximates function space distance (FSD) for continual learning regularization by propagating input distribution moments through a step-wise linearized model, without storing any actual datapoints.

## Setup

Requires Python 3.10+. Clone the repo and install dependencies via [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/nikitadhawan/liftr
cd liftr
uv sync
uv pip install -e .
source .venv/bin/activate
```

## Experiments

The two main entry points are:

- `main.py` trains a model with FSD regularization across a sequence of tasks.
- `compare_fsd.py` compares a given FSD estimator against the Oracle estimator.

Example:

```bash
python main.py \
    dataset=arithmetic \
    model=transformer \
    model.d_model=512 \
    model.d_hidden=1024 \
    model.num_heads=4 \
    model.num_blocks=2 \
    model.output_shape=115 \
    optimizer=adamw \
    learning_rate=3e-4 \
    batch_size=256 \
    train_epochs=200 \
    fsd_estimator=liftr \
    fsd_estimator.stochastic=True \
    fsd_weight=1
```

## Project structure

```
src/
  datasets/       # Arithmetic continual learning benchmark
  models/         # Transformer implementation
  fsd_estimators/ # LIFTR, EWC, NTK, RandomSubset, GroundTruth
  liftr_modes/    # Per-layer moment propagation rules
conf/             # Hydra config files
main.py           # Training entry point
compare_fsd.py    # FSD estimator comparison
```
