"""
Discrete Wavelet Transform (DWT) implementation for image forgery detection
"""

# Import key functions to make them available directly from the dwt package
from .utils import extract_dwt_features, load_model, save_model, load_dataset
from .train import train_model, save_report
from .test import detect_forgery

# Define what gets imported with "from dwt import *"
__all__ = [
    'extract_dwt_features',
    'load_model',
    'save_model',
    'load_dataset',
    'train_model',
    'save_report',
    'detect_forgery'
]