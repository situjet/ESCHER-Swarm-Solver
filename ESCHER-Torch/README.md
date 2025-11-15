# ESCHER-Torch

PyTorch reimplementation of the ESCHER algorithm for imperfect-information
games. This variant is intended for WSL Ubuntu 22.04 environments configured
with PyTorch and CUDA 12.6. The solver targets standard OpenSpiel games, and
has been validated on Leduc poker.

## Requirements

* Python 3.10+
* PyTorch with CUDA support (optional, CPU fallback available)
* OpenSpiel (Python bindings)

## Running on Leduc Poker

```bash
python run_escher_torch_leduc.py
```

Results (exploitability checkpoints, node counts, etc.) are stored under
`results/leduc/`.
