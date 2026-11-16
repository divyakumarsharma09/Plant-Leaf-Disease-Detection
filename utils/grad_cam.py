"""
Grad-CAM (Gradient-weighted Class Activation Mapping).

Reference: Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391

Given a trained model and an input image, produces a heatmap highlighting
the regions the model attended to when making its prediction.
"""
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

import config
