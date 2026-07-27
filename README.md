# PCG - FabricPC Predictive Coding Experiments

External nodes and experiments for [FabricPC](https://github.com/pderp/fabricpc) predictive coding networks.

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. Clone FabricPC

```bash
git clone https://github.com/pderp/fabricpc.git
cd fabricpc
pip install -e ".[all]"
cd ..
```

**Backend selection:** The `[all]` extra installs dev, tfds, experiments, and viz dependencies.
For GPU support, append a backend extra, e.g. `pip install -e ".[all,cuda12]"` or `pip install -e ".[all,cuda13]"`.
For CPU-only, use `pip install -e ".[all,cpu]"`.

### 3. Clone this repo

```bash
git clone https://github.com/pderp/pcg.git
cd pcg
pip install -e .
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Full Dependency List

FabricPC (`pip install -e ".[all]"`) installs:

**Core:**
- `jax`, `jaxlib`
- `optax>=0.1.7`
- `orbax-checkpoint>=0.4.0`
- `flax>=0.7.5`
- `chex>=0.1.84`
- `jaxtyping>=0.2.23`
- `numpy>=1.24.0`
- `tqdm>=4.65.0`
- `optuna`

**Dev:**
- `pytest>=7.0.0`, `hypothesis>=6.0.0`
- `black[colorama]==26.1.0`, `ruff==0.15.19`, `mypy>=1.0.0`
- `pre-commit>=3.0.0`

**TFDS:**
- `tensorflow-datasets>=4.9.0`, `tensorflow>=2.15.0`
- `importlib_resources`, `tokenizers>=0.15.0`

**Experiments:**
- `scipy>=1.10.0`

**Viz:**
- `plotly>=5.0.0`, `kaleido>=0.2.1`, `pandas>=2.0.0`
- `aim>=3.0.0` (excluded on Windows / Python 3.13+)

## Structure

- `src/pcg_nodes/` - External FabricPC nodes (LinearResidual)
- `tests/` - Unit tests
- `docs/dev-plans/` - Development plans

## Requirements

- Python 3.10+
- JAX
- FabricPC
