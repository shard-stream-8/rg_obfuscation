import importlib
import importlib.util
import sys
import os

TASK_REGISTRY = {
    "leg_counting": {
        "config_class": "reasoning_gym.arithmetic.leg_counting.LegCountingConfig",
        "dataset_class": "reasoning_gym.arithmetic.leg_counting.LegCountingDataset"
    },
    # Add more tasks here as needed
}

def _import_class(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_task(task_name, custom_verifier_path=None):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task '{task_name}' not found in TASK_REGISTRY. Available: {list(TASK_REGISTRY.keys())}")
    entry = TASK_REGISTRY[task_name]
    ConfigClass = _import_class(entry["config_class"])
    DatasetClass = _import_class(entry["dataset_class"])
    config = ConfigClass()
    if custom_verifier_path:
        spec = importlib.util.spec_from_file_location("custom_verifier", custom_verifier_path)
        custom_verifier = importlib.util.module_from_spec(spec)
        sys.modules["custom_verifier"] = custom_verifier
        spec.loader.exec_module(custom_verifier)
        return DatasetClass(config, verifier=custom_verifier.verifier)
    else:
        return DatasetClass(config) 