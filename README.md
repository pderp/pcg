# PCG - FabricPC Predictive Coding Experiments

External nodes and experiments for [FabricPC](https://github.com/yourorg/fabricpc) predictive coding networks.

## Setup

### 1. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. Clone FabricPC
```bash
git clone https://github.com/yourorg/fabricpc.git
cd fabricpc
pip install -e .
cd ..
```

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

## Structure
- `src/pcg_nodes/` - External FabricPC nodes (LinearResidual)
- `tests/` - Unit tests
- `docs/dev-plans/` - Development plans

## Requirements
- Python 3.10+
- JAX
- FabricPC