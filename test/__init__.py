"""
Test modules for image forgery detection system
"""

# Import main test functions for easier access
from .test_forgery_detection_comparative import ModelTester, DWTModelTester, DyWTModelTester, RDLNNModelTester, ComparativeTester

# Define what gets imported with "from test import *"
__all__ = [
    'ModelTester',
    'DWTModelTester',
    'DyWTModelTester', 
    'RDLNNModelTester',
    'ComparativeTester'
]