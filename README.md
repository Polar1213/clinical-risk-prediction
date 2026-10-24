# Clinical Risk Prediction — Baseline Prototype

A small, reproducible **educational** machine-learning pipeline that demonstrates how to preprocess tabular health-related features and evaluate a binary classifier.

**Important:** All records and labels are synthetically generated from an invented formula. The model does **not** predict real cardiovascular risk, is **not** clinically validated, and must not be used for medical decisions. Any evaluation scores measure only performance on the artificial dataset.

## What it demonstrates
- Synthetic dataset generation with missing measurements
- Stratified train/test split
- Median imputation and feature scaling fitted on training data only (prevents test-set leakage)
- Logistic regression baseline with scikit-learn Pipeline
- Test ROC AUC, classification report, ROC curve, and confusion matrix

## Run locally

1. Install Python 3.10+ and open this folder in VS Code.
2. In the VS Code terminal, run:

```bash
python -m pip install -r requirements.txt
python train.py
```

On some Windows setups, use `py` instead of `python`.

The program prints evaluation metrics and creates `outputs/roc_curve.png` and `outputs/confusion_matrix.png`.

## Next steps
Replace synthetic data with a suitable, legally usable public dataset; document its source and limitations; compare logistic regression with a tree-based model and PyTorch neural network; evaluate calibration and subgroup performance.
