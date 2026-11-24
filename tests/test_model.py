"""Unit tests. Run with: pytest tests/ -v"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from disease_info import DISEASE_INFO, get_info
from utils.preprocessing import preprocess_pil, is_allowed_file


def test_class_count():
    assert len(config.CLASS_NAMES) == config.NUM_CLASSES == 38


def test_disease_info_covers_all_classes():
    """Every class in config.CLASS_NAMES must have an entry in DISEASE_INFO."""
    missing = [c for c in config.CLASS_NAMES if c not in DISEASE_INFO]
    assert not missing, f"Missing disease info for: {missing}"


def test_disease_info_fields():
    for cls, info in DISEASE_INFO.items():
        assert "cause" in info, cls
        assert "symptoms" in info, cls
        assert "treatment" in info, cls
        assert len(info["cause"]) > 0
        assert len(info["treatment"]) > 0
