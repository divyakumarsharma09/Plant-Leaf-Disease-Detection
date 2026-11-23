"""Unit tests. Run with: pytest tests/ -v"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from disease_info import DISEASE_INFO, get_info
from utils.preprocessing import preprocess_pil, is_allowed_file
