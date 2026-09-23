import os
import sys
import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model
from src.model_identity import compute_model_id, _library_versions

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
RAW_DATA_PATH = os.path.join(_PROJECT_ROOT, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')

CONFIG = {
    'model_type': 'GradientBoostingClassifier',
    'n_estimators': 100,
    'random_state': 42,
}

from decimal import Decimal, ROUND_HALF_UP

METRIC_DECIMALS = {
    'roc_auc': 3,
    'precision': 2,
    'recall': 2,
    'f1': 2,
}

def _quantize(value, decimals):
    quantizer = Decimal('1.' + '0' * decimals)
    return float(Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP))

def _load_notebook_metrics(notebook_path, model_name='Gradient Boosting'):
    with open(notebook_path, encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            text = ''.join(out.get('text', []))
            if model_name in text and 'roc_auc' in text:
                for line in text.splitlines():
                    if line.strip().startswith(model_name):
                        nums = []
                        for token in line.split():
                            try:
                                nums.append(float(token))
                            except ValueError:
                                pass
                        if len(nums) >= 4:
                            roc_auc, precision, recall, f1 = nums[:4]
                            return {
                                'roc_auc': roc_auc,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                            }
    raise ValueError(f"Could not find recorded metrics for {model_name} in {notebook_path}")
    

_recorded = _load_notebook_metrics(os.path.join(_PROJECT_ROOT, 'notebooks', 'notebook03_modeling.ipynb'))

PUBLISHED_METRICS = {
    key: _quantize(value, METRIC_DECIMALS[key])
    for key, value in _recorded.items()
    if key in METRIC_DECIMALS
}

def check_against_published(metrics):
    mismatches = []
    for key, published in PUBLISHED_METRICS.items():
        actual = metrics[key]
        decimals = METRIC_DECIMALS[key]
        quantizer = Decimal('1.' + '0' * decimals) if decimals > 0 else Decimal('1')
        actual_rounded = float(Decimal(str(actual)).quantize(quantizer, rounding=ROUND_HALF_UP))
        if actual_rounded != published:
            mismatches.append(
                f"{key}: got {actual:.4f} (rounds to {actual_rounded}), "
                f"expected {published}"
            )
    if mismatches:
        raise AssertionError(
            "Rebuilt model does not match published metrics:\n" + "\n".join(mismatches)
        )
    print("✓ Metrics match published results (rounded to published precision).")

def main():
    df = pd.read_csv(RAW_DATA_PATH)
    X_train_res, X_test, y_train_res, y_test, feature_names = preprocess_data(df)

    model = GradientBoostingClassifier(
        n_estimators=CONFIG['n_estimators'],
        random_state=CONFIG['random_state']
    )
    model.fit(X_train_res, y_train_res)

    metrics = evaluate_model(model, X_test, y_test, model_name=CONFIG['model_type'])
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    check_against_published(metrics)

    code_paths = code_paths = [
    os.path.join(_PROJECT_ROOT, 'src', 'preprocessing.py'),
    os.path.join(_PROJECT_ROOT, 'src', 'evaluation.py'),
    os.path.join(_PROJECT_ROOT, 'src', 'train.py'),
]
    model_id = compute_model_id(RAW_DATA_PATH, code_paths, CONFIG)

    model_dir = os.path.join(MODELS_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))

    metadata = {
        'model_id': model_id,
        'model_type': CONFIG['model_type'],
        'config': CONFIG,
        'features': feature_names,
        'metrics': metrics,
        'library_versions': _library_versions(),
    }
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model with identity: {model_id}")
    print(f"Location: {model_dir}")

if __name__ == '__main__':
    main()