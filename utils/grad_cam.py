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
def _find_last_conv_layer(model):
    """Walk the model backwards and return the last Conv2D layer's name.
    Handles wrapped Keras Application backbones (Sequential / Functional)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If layer is itself a Model (e.g. MobileNetV2 backbone), recurse.
        if hasattr(layer, "layers"):
            for inner in reversed(layer.layers):
                if isinstance(inner, tf.keras.layers.Conv2D):
                    # Return the *sub-model* layer name; caller handles that.
                    return layer.name
    raise ValueError("No Conv2D layer found in model.")
