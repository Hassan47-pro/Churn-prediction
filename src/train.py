import os
import sys
import json
import subprocess
import joblib
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model
from src.model_identity import compute_model_id, _library_versions


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


CONFIG = {
    "model_type": "GradientBoostingClassifier",
    "n_estimators": 100,
    "random_state": 42,
}


# This is the pre-existing notebook version from main.
# It is deliberately fixed so the acceptance gate cannot move
# when somebody re-runs or edits the working notebook.
REFERENCE_COMMIT = "17b363857cbad2c7a90411be5d34b94660f008b2"
REFERENCE_NOTEBOOK = "notebooks/notebook03_modeling.ipynb"


METRIC_DECIMALS = {
    "roc_auc": 3,
    "precision": 2,
    "recall": 2,
    "f1": 2,
}


def _quantize(value, decimals):
    quantizer = Decimal("1." + "0" * decimals)
    return float(
        Decimal(str(value)).quantize(
            quantizer,
            rounding=ROUND_HALF_UP,
        )
    )


def _load_reference_notebook():
    """Load the pre-existing notebook directly from its fixed Git commit."""
    result = subprocess.run(
        [
            "git",
            "show",
            f"{REFERENCE_COMMIT}:{REFERENCE_NOTEBOOK}",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def _load_notebook_metrics(model_name="Gradient Boosting"):
    """Extract recorded metrics from the immutable reference notebook."""
    nb = _load_reference_notebook()

    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", []))

            if model_name in text and "roc_auc" in text:
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
                                "roc_auc": roc_auc,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                            }

    raise ValueError(
        f"Could not find recorded metrics for {model_name} "
        f"in {REFERENCE_COMMIT}:{REFERENCE_NOTEBOOK}"
    )


_recorded = _load_notebook_metrics()

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

        quantizer = (
            Decimal("1." + "0" * decimals)
            if decimals > 0
            else Decimal("1")
        )

        actual_rounded = float(
            Decimal(str(actual)).quantize(
                quantizer,
                rounding=ROUND_HALF_UP,
            )
        )

        if actual_rounded != published:
            mismatches.append(
                f"{key}: got {actual:.4f} "
                f"(rounds to {actual_rounded}), "
                f"expected {published}"
            )

    if mismatches:
        raise AssertionError(
            "Rebuilt model does not match published metrics:\n"
            + "\n".join(mismatches)
        )

    print("✓ Metrics match published results "
          "(rounded to published precision).")


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    X_train_res, X_test, y_train_res, y_test, feature_names = preprocess_data(df)

    model = GradientBoostingClassifier(
        n_estimators=CONFIG["n_estimators"],
        random_state=CONFIG["random_state"],
    )

    model.fit(X_train_res, y_train_res)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        model_name=CONFIG["model_type"],
    )

    print("Evaluation metrics:")

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    check_against_published(metrics)

    code_paths = [
        os.path.join(PROJECT_ROOT, "src", "preprocessing.py"),
        os.path.join(PROJECT_ROOT, "src", "evaluation.py"),
        os.path.join(PROJECT_ROOT, "src", "train.py"),
    ]

    model_id = compute_model_id(
        RAW_DATA_PATH,
        code_paths,
        CONFIG,
    )

    model_dir = os.path.join(
        MODELS_DIR,
        model_id,
    )

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(
        model,
        os.path.join(model_dir, "model.pkl"),
    )

    metadata = {
        "model_id": model_id,
        "model_type": CONFIG["model_type"],
        "config": CONFIG,
        "features": feature_names,
        "metrics": metrics,
        "library_versions": _library_versions(),
    }

    with open(
        os.path.join(model_dir, "metadata.json"),
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model with identity: {model_id}")
    print(f"Location: {model_dir}")


if __name__ == "__main__":
    main()