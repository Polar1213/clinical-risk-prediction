"""Educational cardiovascular risk classification baseline using synthetic data.

The generated data are NOT real patients and results have no clinical meaning.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT = Path('outputs')
FEATURES = ['age', 'systolic_bp', 'cholesterol', 'smoker', 'bmi']


def make_synthetic_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 81, n)
    bp = np.clip(rng.normal(125 + .35 * (age - 50), 17, n), 85, 220)
    cholesterol = np.clip(rng.normal(200 + .3 * (age - 50), 38, n), 100, 350)
    smoker = rng.binomial(1, .25, n)
    bmi = np.clip(rng.normal(27, 5, n), 16, 48)
    # Artificial data-generating rule, not a medical risk formula.
    log_odds = (-2.0 + .045 * (age - 50) + .028 * (bp - 125)
                + .012 * (cholesterol - 200) + .85 * smoker + .07 * (bmi - 27))
    probability = 1 / (1 + np.exp(-log_odds))
    target = rng.binomial(1, probability)
    df = pd.DataFrame({'age': age, 'systolic_bp': bp, 'cholesterol': cholesterol,
                       'smoker': smoker, 'bmi': bmi, 'risk_label': target})
    # Simulate missing measurements to demonstrate train-only imputation.
    for col in ['systolic_bp', 'cholesterol', 'bmi']:
        missing = rng.choice(n, size=int(n * .04), replace=False)
        df.loc[missing, col] = np.nan
    return df


def main():
    OUTPUT.mkdir(exist_ok=True)
    df = make_synthetic_data()
    X, y = df[FEATURES], df['risk_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y)

    numeric = FEATURES
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric)
    ])
    model = Pipeline([
        ('preprocessing', preprocessing),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    print(f'Synthetic dataset: {len(df)} rows; test set: {len(y_test)} rows')
    print(f'Test ROC AUC: {auc:.3f}')
    print(classification_report(y_test, predicted, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, scores, ax=ax)
    ax.set_title('ROC curve — synthetic data (not clinical)')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'roc_curve.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, ax=ax, colorbar=False)
    ax.set_title('Confusion matrix — synthetic data')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    print('Saved plots in outputs/')


if __name__ == '__main__':
    main()
