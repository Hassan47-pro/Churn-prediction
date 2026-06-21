import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate a fitted model on the test set.
    Returns a dict of metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'model': model_name,
        'roc_auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    return metrics 



def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for multiple fitted models.
    models_dict: {name: fitted_model}
    """
    plt.figure(figsize=(8, 6))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curves.png')
    plt.show()
