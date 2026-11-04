"""
src/models module.
Exports CrossModalFramingModel and baseline ablation architectures.
"""

from src.models.fusion_model import CrossModalFramingModel
from src.models.baselines import (
    TextOnlyModel,
    VisionGlobalModel,
    VisionRegionModel,
    LateFusionModel,
)

__all__ = [
    "CrossModalFramingModel",
    "TextOnlyModel",
    "VisionGlobalModel",
    "VisionRegionModel",
    "LateFusionModel",
]
