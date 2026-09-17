import os
import sys
import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model
from src.model_identity import compute_model_id

RAW_DATA_PATH = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODELS_DIR = 'models'

CONFIG = {
    'model_type': 'GradientBoostingClassifier',
    'n_estimators': 100,
    'random_state': 42,
}

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

    code_paths = ['src/preprocessing.py', 'src/evaluation.py', 'src/train.py']
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
    }
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model with identity: {model_id}")
    print(f"Location: {model_dir}")

if __name__ == '__main__':
    main()