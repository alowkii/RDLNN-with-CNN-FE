"""
Image Forgery Detection System - Module initialization
Provides exports for the main project modules
"""

# Import main classes and functions to make them available directly from the package
from .rdlnn import RegressionDLNN
from .preprocessing import preprocess_image
from .utils import logger, plot_training_history, setup_logging, setup_signal_handlers

# Define what gets imported with "from modules import *"
__all__ = [
    'RegressionDLNN',
    'preprocess_image',
    'logger',
    'plot_training_history',
    'setup_logging',
    'setup_signal_handlers'
]

# Version information
__version__ = '1.0.0'