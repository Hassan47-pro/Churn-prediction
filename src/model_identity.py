import hashlib
import json
import importlib.metadata

def _hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _library_versions():
    packages = ['scikit-learn', 'imbalanced-learn', 'pandas', 'numpy']
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = 'unknown'
    return versions

def compute_model_id(data_path, code_paths, config):
    """
    Deterministic model identity derived from the data, code, config, and
    library versions that produced it. Same inputs -> same ID. Change any
    input -> different ID.
    """
    hasher = hashlib.sha256()
    hasher.update(_hash_file(data_path).encode())
    for path in sorted(code_paths):
        hasher.update(_hash_file(path).encode())
    hasher.update(json.dumps(config, sort_keys=True).encode())
    hasher.update(json.dumps(_library_versions(), sort_keys=True).encode())
    digest = hasher.hexdigest()[:12]
    return f"churn-{digest}"


def load_model(model_id, models_dir='models'):
    import os, joblib
    model_path = os.path.join(models_dir, model_id, 'model.pkl')
    metadata_path = os.path.join(models_dir, model_id, 'metadata.json')
    with open(metadata_path) as f:
        metadata = json.load(f)
    model = joblib.load(model_path)
    return model, metadata