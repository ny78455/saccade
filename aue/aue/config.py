"""
aue/aue/config.py
Configuration loader for AUE.

Reads config.default.yaml (or a user-supplied override) and returns AUEConfig.
Same pattern as ASVL's config.py.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from .types import AUEConfig, _assert_importance_weights_sum

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.default.yaml"


def load_config(path: Optional[str] = None) -> AUEConfig:
    """
    Load AUEConfig from a YAML file.

    Args:
        path: Path to a YAML config file. If None, loads config.default.yaml
              from the aue/ package root.

    Returns:
        AUEConfig instance with all values populated.

    Raises:
        AssertionError if importance_weights don't sum to ~1.0.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning(
            "AUE: config file not found at %s — using dataclass defaults.",
            config_path,
        )
        cfg = AUEConfig()
        _assert_importance_weights_sum(cfg)
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Flatten importance_weights if present
    weights = raw.pop("importance_weights", None)

    cfg = AUEConfig(**{k: v for k, v in raw.items() if hasattr(AUEConfig, k) or k in AUEConfig.__dataclass_fields__})

    if weights is not None:
        cfg.importance_weights = weights

    _assert_importance_weights_sum(cfg)
    logger.info("AUE: config loaded from %s", config_path)
    return cfg
