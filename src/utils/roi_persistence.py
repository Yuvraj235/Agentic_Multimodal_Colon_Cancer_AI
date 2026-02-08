import json
import os
import numpy as np


def _to_python(obj):
    """
    Recursively convert NumPy / Torch types to native Python types
    so they are JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_python(v) for v in obj]

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def save_metrics(metrics, localization):
    os.makedirs("outputs/json", exist_ok=True)

    payload = {
        "metrics": _to_python(metrics),
        "localization": _to_python(localization)
    }

    with open("outputs/json/roi_metrics.json", "w") as f:
        json.dump(payload, f, indent=4)