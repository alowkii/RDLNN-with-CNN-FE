"""
RDLNN with CNN-FE: Image Forgery Detection System

This package provides a comprehensive framework for detecting image forgeries 
using Regression Deep Learning Neural Networks (RDLNN) with Convolutional Neural 
Network Feature Extraction (CNN-FE).
"""

# Make the key modules available at the package level
from modules import *
import dwt
import dywt
import tools
import training

# Define the version
__version__ = '1.0.0'