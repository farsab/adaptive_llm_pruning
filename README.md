
# Adaptive LLM Pruning
This repository demonstrates **adaptive pruning** and **reactivation bursts** on a small language model (DistilGPT-2) using PyTorch and Hugging Face.

## Key Features
- Adaptive pruning based on gradient variance.
- Periodic reactivation bursts to restore alignment stability.
- Calibration metrics (ECE, NLL).
- Interpretability proxy via attention entropy.
- Matplotlib visualizations of metrics over time.

## Usage
```bash
python main.py --epochs 2 --target_sparsity 0.4 --reactivate_every 200
```

Results are saved under `results/` as plots and CSV logs.

