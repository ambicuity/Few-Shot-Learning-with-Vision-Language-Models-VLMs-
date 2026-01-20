# Reproducibility Checklist

## Code
- [x] All training code is provided in `code/train.py`.
- [x] Dependencies are listed in `code/requirements.txt`.
- [x] Seeds are fixed (set_seed function) for deterministic results (Seeds 1, 2, 3).
- [x] Hyperparameters (shots=16, epochs=50) are explicitly defined.

## Data
- [x] Datasets (MVTec, EuroSAT, Oxford Pets) are public.
- [x] Data loading logic (simulated for GHA, real paths needed) is documented.

## Experiments
- [x] GitHub Actions workflow `.github/workflows/experiments.yml` automates the run.
- [x] No local execution required.

## Results
- [x] Mean and Standard Deviation reported over 3 seeds.
- [x] Comparison with 3 baseline methods (Zero-Shot, CoOp, Tip-Adapter).
