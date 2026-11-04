"""
scripts package.
Exposes pipeline entry points.
"""

import importlib.util
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_train_file = os.path.join(_script_dir, "03_train.py")

train_pipeline = None

if os.path.exists(_train_file):
    _spec = importlib.util.spec_from_file_location("scripts.03_train", _train_file)
    if _spec and _spec.loader:
        _train_mod = importlib.util.module_from_spec(_spec)
        sys.modules["scripts.03_train"] = _train_mod
        _spec.loader.exec_module(_train_mod)
        train_pipeline = getattr(_train_mod, "train_pipeline", None)

__all__ = ["train_pipeline"]
