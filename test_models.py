#!/usr/bin/env python3
"""
Comprehensive test script for image forgery detection models.
Tests RDLNN, DWT, and DyWT models on the provided test datasets.
Fixed version to handle import issues, tqdm progress bars, and ensure dataset results are properly saved.
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
from PIL import Image
import logging

# Configure CUDA settings for better performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_testing.log')
    ]
)
logger = logging.getLogger('model_testing')
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_tqdm_logging():
    """
    Configure logging to work properly with tqdm
    This ensures logging messages don't break the progress bar
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler that works with tqdm
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Add file handler (regular file handler is fine)
    file_handler = logging.FileHandler('model_testing.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Add the handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set the logging level
    root_logger.setLevel(logging.INFO)
    
    return root_logger

# Try to import modules - with fallbacks for missing modules
try:
    from modules.rdlnn import RegressionDLNN
    from modules.feature_extractor import PDyWTCNNDetector
    RDLNN_AVAILABLE = True
    logger.info("RDLNN module loaded successfully")
except ImportError as e:
    logger.error(f"Error importing RDLNN module: {e}")
    RDLNN_AVAILABLE = False

# Import DWT modules - with custom implementation for extract_dwt_features
DWT_AVAILABLE = False
try:
    # Fix for relative import issues
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dwt'))
    from dwt.utils import load_model as load_dwt_model
    
    # Define our own extract_dwt_features function if import fails
    try:
        from dwt.utils import extract_dwt_features
    except ImportError:
        logger.warning("Could not import extract_dwt_features from dwt.utils, using custom implementation")
        
        # Custom implementation based on the original code
        def extract_dwt_features(image_path):
            """
            Extract Discrete Wavelet Transform features from an image.
            Custom implementation to avoid import issues.
            """
            try:
                import numpy as np
                import cv2
                import pywt
                
                # Read image and convert to grayscale
                img = cv2.imread(image_path)
                if img is None:
                    return None
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply 2D DWT using Haar wavelet
                coeffs = pywt.dwt2(gray, 'haar')
                cA, (cH, cV, cD) = coeffs
                
                # Extract statistical features from each sub-band
                features = []
                
                # Process each coefficient matrix
                for coef in [cA, cH, cV, cD]:
                    # Calculate statistical features
                    mean = np.mean(coef)
                    std = np.std(coef)
                    entropy = np.sum(-coef * np.log2(np.abs(coef) + 1e-10))
                    energy = np.sum(coef ** 2)
                    
                    # Add features to our vector
                    features.extend([mean, std, entropy, energy])
                
                return features
            except Exception as e:
                logger.error(f"Error in extract_dwt_features: {e}")
                return None
    
    DWT_AVAILABLE = True
    logger.info("DWT module loaded successfully")
except ImportError as e:
    logger.error(f"Error importing DWT module: {e}")
    DWT_AVAILABLE = False

# Import DyWT modules - with custom implementation
DYWT_AVAILABLE = False
try:
    # Fix for relative import issues
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dywt'))
    from dywt.utils import load_model as load_dywt_model
    
    # Define our own extract_dyadic_wavelet_features function if import fails
    try:
        from dywt.utils import extract_dyadic_wavelet_features
    except ImportError:
        logger.warning("Could not import extract_dyadic_wavelet_features from dywt.utils, using custom implementation")
        
        # Custom implementation based on the original code
        def extract_dyadic_wavelet_features(image_path, decomp_level=3):
            """
            Extract Dyadic Wavelet Transform features from an image.
            Custom implementation to avoid import issues.
            """
            try:
                import numpy as np
                import cv2
                import pywt
                
                # Read image and convert to grayscale
                img = cv2.imread(image_path)
                if img is None:
                    return None
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply multilevel 2D DWT using Daubechies wavelet
                coeffs = pywt.wavedec2(gray, 'db4', level=decomp_level)
                
                # Extract statistical features from each sub-band
                features = []
                
                # Calculate statistical features for a coefficient matrix
                def extract_stats(coef):
                    # Calculate statistical features
                    mean = np.mean(coef)
                    std = np.std(coef)
                    skewness = np.mean(((coef - mean) / (std + 1e-10)) ** 3) if std > 0 else 0
                    kurtosis = np.mean(((coef - mean) / (std + 1e-10)) ** 4) - 3 if std > 0 else 0
                    entropy = -np.sum((np.abs(coef) + 1e-10) * np.log2(np.abs(coef) + 1e-10))
                    energy = np.sum(coef ** 2)
                    
                    return [mean, std, skewness, kurtosis, entropy, energy]
                
                # Process approximation coefficient matrix (cA)
                cA = coeffs[0]
                features.extend(extract_stats(cA))
                
                # Process detail coefficient matrices (cH, cV, cD) for each level
                for detail_coeffs in coeffs[1:]:
                    cH, cV, cD = detail_coeffs
                    features.extend(extract_stats(cH))
                    features.extend(extract_stats(cV))
                    features.extend(extract_stats(cD))
                
                return features
            except Exception as e:
                logger.error(f"Error in extract_dyadic_wavelet_features: {e}")
                return None
    
    DYWT_AVAILABLE = True
    logger.info("DyWT module loaded successfully")
except ImportError as e:
    logger.error(f"Error importing DyWT module: {e}")
    DYWT_AVAILABLE = False

def setup_argument_parser():
    """Set up and parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test image forgery detection models')
    
    # Required arguments
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test datasets')
    
    # Model paths
    parser.add_argument('--rdlnn_model', type=str, default='data/models/rdlnn_model.pth',
                        help='Path to RDLNN model')
    parser.add_argument('--dwt_model', type=str, default='data/models/dwt_forgery_model.pkl',
                        help='Path to DWT model')
    parser.add_argument('--dywt_model', type=str, default='data/models/dyadic_forgery_model.pkl',
                        help='Path to DyWT model')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of results')
    
    # Test parameters
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override classification threshold')
    parser.add_argument('--test_limit', type=int, default=None,
                        help='Limit number of test images per dataset')
    
    # GPU options
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU acceleration if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use')
    
    return parser.parse_args()

def test_rdlnn_model(model, dataset_dir, output_dir, threshold=None, limit=None, device=None):
    """
    Test RDLNN model on a dataset
    
    Args:
        model: Loaded RDLNN model
        dataset_dir: Directory containing test images
        output_dir: Directory to save results
        threshold: Classification threshold (uses model's threshold if None)
        limit: Maximum number of images to test (tests all if None)
        device: Device to use for processing (GPU/CPU)
    
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing RDLNN model on {dataset_dir}")
    
    # Use model's threshold if not overridden
    if threshold is None and hasattr(model, 'threshold'):
        threshold = model.threshold
    elif threshold is None:
        threshold = 0.675
    
    logger.info(f"Using threshold: {threshold}")
    
    # Initialize detector for feature extraction
    detector = PDyWTCNNDetector()
    
    # Move detector to device if provided
    if device is not None and device.type == 'cuda':
        detector.device = device
        if hasattr(detector, 'pdywt'):
            detector.pdywt.device = device
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        image_files.extend(list(Path(dataset_dir).glob(f"*{ext}")))
        image_files.extend(list(Path(dataset_dir).glob(f"*{ext.upper()}")))
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} test images")
    
    # Initialize result containers
    results = {
        'predictions': [],
        'probabilities': [],
        'filenames': [],
        'processing_times': []
    }
    
    # Initialize progress bar
    # Use a format that doesn't get broken by log messages
    dataset_name = os.path.basename(dataset_dir)
    pbar = tqdm(
        total=len(image_files), 
        desc=f"Processing {dataset_name}",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Process images in small batches to optimize GPU usage
    batch_size = 8  # Process 8 images at a time
    
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        for img_path in batch_files:
            try:
                # Use tqdm.write for logging during the loop to avoid breaking the progress bar
                start_time = time.time()
                
                # Extract features using the detector
                feature_vector = detector.extract_features(str(img_path))
                
                if feature_vector is None:
                    # Use tqdm.write instead of logger.warning
                    tqdm.write(f"Failed to extract features from {img_path}")
                    pbar.update(1)  # Update progress even for failures
                    continue
                    
                # Reshape to 2D array for prediction
                feature_vector = feature_vector.reshape(1, -1)
                
                # Apply feature selection if available in the model
                if hasattr(model, 'feature_selector') and model.feature_selector is not None:
                    try:
                        feature_vector = model.feature_selector(feature_vector)
                    except Exception as e:
                        tqdm.write(f"Feature selection failed: {e}")
                
                # Get predictions
                predictions, probabilities = model.predict(feature_vector)
                
                # Store results
                results['predictions'].append(predictions[0])
                results['probabilities'].append(probabilities[0])
                results['filenames'].append(os.path.basename(img_path))
                results['processing_times'].append(time.time() - start_time)
                
                # Calculate prediction status string
                pred_status = "FORGED" if predictions[0] == 1 else "AUTHENTIC"
                pred_color = "\033[91m" if predictions[0] == 1 else "\033[92m"  # Red for forged, green for authentic
                
                # Update progress bar with additional info (no longer shown in postfix)
                proc_time = time.time() - start_time
                
                # Update the progress bar
                pbar.update(1)
                
                # After updating the progress bar, log the result without breaking the bar
                tqdm.write(f"{os.path.basename(img_path)}: {pred_color}{pred_status}\033[0m (p={probabilities[0]:.4f}, t={proc_time:.2f}s)")
                
            except Exception as e:
                tqdm.write(f"Error processing {img_path}: {e}")
                pbar.update(1)  # Update progress even for errors
        
        # Clear GPU cache after each batch if using GPU
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Close the progress bar
    pbar.close()
    
    # Calculate statistics
    results['accuracy'] = calculate_accuracy(results, dataset_dir)
    results['avg_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0
    
    # Log summary
    logger.info(f"Completed testing on {dataset_dir}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}, Avg time: {results['avg_time']:.4f}s")
    
    # Save results
    save_results(results, dataset_dir, output_dir, "rdlnn")
    
    return results

def test_dwt_model(model, dataset_dir, output_dir, threshold=0.5, limit=None, device=None):
    """
    Test DWT model on a dataset
    
    Args:
        model: Loaded DWT model
        dataset_dir: Directory containing test images
        output_dir: Directory to save results
        threshold: Classification threshold
        limit: Maximum number of images to test
        device: Device to use for processing (GPU/CPU) - not directly used for DWT but kept for consistency
    
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing DWT model on {dataset_dir}")
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        image_files.extend(list(Path(dataset_dir).glob(f"*{ext}")))
        image_files.extend(list(Path(dataset_dir).glob(f"*{ext.upper()}")))
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} test images")
    
    # Initialize result containers
    results = {
        'predictions': [],
        'filenames': [],
        'processing_times': []
    }
    
    # Initialize progress bar
    # Use a format that doesn't get broken by log messages
    dataset_name = os.path.basename(dataset_dir)
    pbar = tqdm(
        total=len(image_files), 
        desc=f"Processing {dataset_name}",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Process each image
    for img_path in image_files:
        try:
            start_time = time.time()
            
            # Extract DWT features
            features = extract_dwt_features(str(img_path))
            
            if features is None:
                tqdm.write(f"Failed to extract features from {img_path}")
                pbar.update(1)  # Update progress even for failures
                continue
            
            # Make prediction with the model
            prediction = model.predict([features])[0]
            
            # Store results
            results['predictions'].append(prediction)
            results['filenames'].append(os.path.basename(img_path))
            results['processing_times'].append(time.time() - start_time)
            
            # Calculate prediction status string
            pred_status = "FORGED" if prediction == 1 else "AUTHENTIC"
            pred_color = "\033[91m" if prediction == 1 else "\033[92m"  # Red for forged, green for authentic
            
            # Update the progress bar
            pbar.update(1)
            
            # After updating the progress bar, log the result without breaking the bar
            proc_time = time.time() - start_time
            tqdm.write(f"{os.path.basename(img_path)}: {pred_color}{pred_status}\033[0m (t={proc_time:.2f}s)")
            
        except Exception as e:
            tqdm.write(f"Error processing {img_path}: {e}")
            pbar.update(1)  # Update progress even for errors
    
    # Close the progress bar
    pbar.close()
    
    # Calculate statistics
    results['accuracy'] = calculate_accuracy(results, dataset_dir)
    results['avg_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0
    
    # Log summary
    logger.info(f"Completed testing on {dataset_dir}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}, Avg time: {results['avg_time']:.4f}s")
    
    # Save results
    save_results(results, dataset_dir, output_dir, "dwt")
    
    return results

def test_dywt_model(model, dataset_dir, output_dir, threshold=0.5, limit=None, device=None):
    """
    Test DyWT model on a dataset
    
    Args:
        model: Loaded DyWT model
        dataset_dir: Directory containing test images
        output_dir: Directory to save results
        threshold: Classification threshold
        limit: Maximum number of images to test
        device: Device to use for processing (GPU/CPU) - not directly used for DyWT but kept for consistency
    
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing DyWT model on {dataset_dir}")
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        image_files.extend(list(Path(dataset_dir).glob(f"*{ext}")))
        image_files.extend(list(Path(dataset_dir).glob(f"*{ext.upper()}")))
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} test images")
    
    # Initialize result containers
    results = {
        'predictions': [],
        'probabilities': [],
        'filenames': [],
        'processing_times': []
    }
    
    # Initialize progress bar
    # Use a format that doesn't get broken by log messages
    dataset_name = os.path.basename(dataset_dir)
    pbar = tqdm(
        total=len(image_files), 
        desc=f"Processing {dataset_name}",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Process each image
    for img_path in image_files:
        try:
            start_time = time.time()
            
            # Extract DyWT features
            features = extract_dyadic_wavelet_features(str(img_path))
            
            if features is None:
                tqdm.write(f"Failed to extract features from {img_path}")
                pbar.update(1)  # Update progress even for failures
                continue
            
            # Make prediction with the model
            # Check if model is dictionary with model and scaler
            if isinstance(model, dict) and 'model' in model and 'scaler' in model:
                # Scale features
                features_scaled = model['scaler'].transform([features])
                
                # Get prediction and probability
                prediction_prob = model['model'].predict_proba(features_scaled)[0, 1]
                prediction = 1 if prediction_prob >= threshold else 0
                
                # Store result with probability
                results['probabilities'].append(prediction_prob)
            else:
                # Legacy model without probability
                prediction = model.predict([features])[0]
                results['probabilities'].append(float(prediction))
            
            # Store results
            results['predictions'].append(prediction)
            results['filenames'].append(os.path.basename(img_path))
            results['processing_times'].append(time.time() - start_time)
            
            # Calculate prediction status string
            pred_status = "FORGED" if prediction == 1 else "AUTHENTIC"
            pred_color = "\033[91m" if prediction == 1 else "\033[92m"  # Red for forged, green for authentic
            
            # Update the progress bar
            pbar.update(1)
            
            # After updating the progress bar, log the result without breaking the bar
            proc_time = time.time() - start_time
            
            # Check if we have probability information to display
            if 'probabilities' in results and len(results['probabilities']) > 0:
                prob = results['probabilities'][-1]
                tqdm.write(f"{os.path.basename(img_path)}: {pred_color}{pred_status}\033[0m (p={prob:.4f}, t={proc_time:.2f}s)")
            else:
                tqdm.write(f"{os.path.basename(img_path)}: {pred_color}{pred_status}\033[0m (t={proc_time:.2f}s)")
            
        except Exception as e:
            tqdm.write(f"Error processing {img_path}: {e}")
            pbar.update(1)  # Update progress even for errors
    
    # Close the progress bar
    pbar.close()
    
    # Calculate statistics
    results['accuracy'] = calculate_accuracy(results, dataset_dir)
    results['avg_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0
    
    # Log summary
    logger.info(f"Completed testing on {dataset_dir}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}, Avg time: {results['avg_time']:.4f}s")
    
    # Save results
    save_results(results, dataset_dir, output_dir, "dywt")
    
    return results

def calculate_accuracy(results, dataset_dir):
    """
    Calculate accuracy based on dataset directory name
    Handles both direct and nested directory structures
    
    Args:
        results: Dictionary with test results
        dataset_dir: Directory containing test images
    
    Returns:
        Accuracy value
    """
    # Determine expected label from directory name
    dir_name = os.path.basename(dataset_dir).lower()
    
    # Check for nested structure (dataset/authentic or dataset/forged)
    parent_dir = os.path.dirname(dataset_dir)
    if os.path.exists(os.path.join(parent_dir, 'authentic')) and os.path.exists(os.path.join(parent_dir, 'forged')):
        # This is a nested structure - use the directory name directly
        expected_label = 1 if 'forged' in dir_name else 0
    else:
        # Try to infer from filenames if available
        if len(results['filenames']) > 0:
            # Check if the first few filenames contain hints about authenticity
            sample_files = results['filenames'][:min(10, len(results['filenames']))]
            forged_count = sum(1 for name in sample_files if 'forged' in name.lower() or 'tp' in name.lower())
            authentic_count = sum(1 for name in sample_files if 'authentic' in name.lower() or 'au' in name.lower())
            
            if forged_count > authentic_count:
                expected_label = 1  # Mostly forged files
            else:
                expected_label = 0  # Mostly authentic files or unknown
        else:
            # Last resort - guess based on directory name
            expected_label = 1 if 'forged' in dir_name or 'tp' in dir_name.lower() else 0
    
    # Calculate accuracy
    if 'predictions' in results and len(results['predictions']) > 0:
        predictions = np.array(results['predictions'])
        accuracy = np.mean(predictions == expected_label)
        return accuracy
    
    return 0.0

def save_results(results, dataset_dir, output_dir, model_name):
    """
    Save test results to file
    
    Args:
        results: Dictionary with test results
        dataset_dir: Directory containing test images
        output_dir: Directory to save results
        model_name: Name of the tested model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dataset name and type (authentic or forged) from path
    # Example path: ./data/test_datasets/columbia/authentic
    path_parts = dataset_dir.split(os.sep)
    
    # Get the dataset name and type
    if len(path_parts) >= 2:
        # The dataset name is the directory before authentic/forged
        # The type is the last directory (authentic or forged)
        dataset_name = path_parts[-2]  # e.g., "columbia"
        dataset_type = path_parts[-1]  # e.g., "authentic" or "forged"
    else:
        # Fallback if path structure doesn't match expectation
        dataset_name = os.path.basename(os.path.dirname(dataset_dir))
        dataset_type = os.path.basename(dataset_dir)
    
    # Create filename following the desired format: {model}_{dataset}_{authentic/forged}_results.csv
    output_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_{dataset_type}_results.csv")
    
    # Write results to CSV
    with open(output_file, 'w') as f:
        # Write header
        header = ['Filename', 'Prediction']
        if 'probabilities' in results:
            header.append('Probability')
        header.append('Processing_Time')
        
        f.write(','.join(header) + '\n')
        
        # Write results
        for i in range(len(results['filenames'])):
            row = [results['filenames'][i], str(results['predictions'][i])]
            if 'probabilities' in results and i < len(results['probabilities']):
                row.append(f"{results['probabilities'][i]:.6f}")
            row.append(f"{results['processing_times'][i]:.6f}")
            
            f.write(','.join(row) + '\n')
    
    logger.info(f"Results saved to {output_file}")
    
    # Also save summary information - append to summary file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'a') as f:
        f.write(f"\n{model_name.upper()} results for {dataset_name} {dataset_type}:\n")
        f.write(f"Total images tested: {len(results['filenames'])}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Average processing time: {results['avg_time']:.4f} seconds\n")
        
        # If probabilities available, add confidence stats
        if 'probabilities' in results and len(results['probabilities']) > 0:
            probs = np.array(results['probabilities'])
            f.write(f"Confidence - Mean: {np.mean(probs):.4f}, Min: {np.min(probs):.4f}, Max: {np.max(probs):.4f}\n")
        
        f.write("-" * 60 + "\n")

def visualize_results(rdlnn_results, dwt_results, dywt_results, output_dir):
    """
    Create visualizations comparing model performance
    
    Args:
        rdlnn_results: Dictionary with RDLNN test results (dataset_dir -> results)
        dwt_results: Dictionary with DWT test results (dataset_dir -> results)
        dywt_results: Dictionary with DyWT test results (dataset_dir -> results)
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect accuracy data
    model_names = []
    dataset_names = []
    dataset_types = []  # "authentic" or "forged"
    accuracy_values = []
    
    # Extract dataset name and type from dataset_dir
    def extract_dataset_info(dataset_dir):
        parts = dataset_dir.split(os.sep)
        if len(parts) >= 2:
            dataset_name = parts[-2]  # e.g., "columbia"
            dataset_type = parts[-1]  # e.g., "authentic" or "forged"
            return dataset_name, dataset_type
        else:
            # Fallback if path structure doesn't match expectation
            return os.path.basename(dataset_dir), "unknown"
    
    # RDLNN results
    for dataset_dir, results in rdlnn_results.items():
        dataset_name, dataset_type = extract_dataset_info(dataset_dir)
        model_names.append('RDLNN')
        dataset_names.append(dataset_name)
        dataset_types.append(dataset_type)
        accuracy_values.append(results['accuracy'])
    
    # DWT results
    for dataset_dir, results in dwt_results.items():
        dataset_name, dataset_type = extract_dataset_info(dataset_dir)
        model_names.append('DWT')
        dataset_names.append(dataset_name)
        dataset_types.append(dataset_type)
        accuracy_values.append(results['accuracy'])
    
    # DyWT results
    for dataset_dir, results in dywt_results.items():
        dataset_name, dataset_type = extract_dataset_info(dataset_dir)
        model_names.append('DyWT')
        dataset_names.append(dataset_name)
        dataset_types.append(dataset_type)
        accuracy_values.append(results['accuracy'])
    
    # Create a bar chart for accuracy comparison
    plt.figure(figsize=(15, 10))
    
    # Group by dataset and type
    unique_dataset_combinations = sorted(set(zip(dataset_names, dataset_types)))
    unique_models = ['RDLNN', 'DWT', 'DyWT']
    
    # Create data for grouped bar chart
    data = {}
    for model in unique_models:
        data[model] = []
        for dataset_name, dataset_type in unique_dataset_combinations:
            # Find indices where model, dataset name and type match
            indices = [i for i, (m, d, t) in enumerate(zip(model_names, dataset_names, dataset_types)) 
                      if m == model and d == dataset_name and t == dataset_type]
            if indices:
                data[model].append(accuracy_values[indices[0]])
            else:
                data[model].append(0)  # No data for this combination
    
    # Plotting
    bar_width = 0.25
    index = np.arange(len(unique_dataset_combinations))
    
    # Plot each model's results
    plt.bar(index - bar_width, data['RDLNN'], bar_width, label='RDLNN', color='blue')
    plt.bar(index, data['DWT'], bar_width, label='DWT', color='green')
    plt.bar(index + bar_width, data['DyWT'], bar_width, label='DyWT', color='red')
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison by Dataset')
    
    # Create more descriptive x-axis labels including the dataset type
    x_labels = [f"{dataset}_{type}" for dataset, type in unique_dataset_combinations]
    plt.xticks(index, x_labels, rotation=45, ha='right')
    
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # Create another plot for processing time comparison
    plt.figure(figsize=(15, 10))
    
    # Collect timing data
    processing_times = {}
    for model in unique_models:
        processing_times[model] = []
        for dataset_name, dataset_type in unique_dataset_combinations:
            # Find which results dictionary to use
            if model == 'RDLNN':
                results_dict = rdlnn_results
            elif model == 'DWT':
                results_dict = dwt_results
            else:  # DyWT
                results_dict = dywt_results
                
            # Find matching dataset
            found = False
            for dir_name, results in results_dict.items():
                dir_dataset, dir_type = extract_dataset_info(dir_name)
                if dir_dataset == dataset_name and dir_type == dataset_type:
                    processing_times[model].append(results['avg_time'])
                    found = True
                    break
                    
            if not found:
                processing_times[model].append(0)  # No data for this dataset
    
    # Plotting
    plt.bar(index - bar_width, processing_times['RDLNN'], bar_width, label='RDLNN', color='blue')
    plt.bar(index, processing_times['DWT'], bar_width, label='DWT', color='green')
    plt.bar(index + bar_width, processing_times['DyWT'], bar_width, label='DyWT', color='red')
    
    plt.xlabel('Dataset')
    plt.ylabel('Average Processing Time (s)')
    plt.title('Model Processing Time Comparison by Dataset')
    plt.xticks(index, x_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'processing_time_comparison.png'))
    plt.close()
    
    # Create separate comparison plots for authentic vs forged data
    for dataset_type in ['authentic', 'forged']:
        # Filter data for this type
        type_indices = [i for i, t in enumerate(dataset_types) if t == dataset_type]
        if not type_indices:
            continue  # Skip if no data for this type
            
        # Create a filtered view of the data
        filtered_models = [model_names[i] for i in type_indices]
        filtered_datasets = [dataset_names[i] for i in type_indices]
        filtered_accuracies = [accuracy_values[i] for i in type_indices]
        
        # Create plot for this type
        plt.figure(figsize=(12, 8))
        
        # Group by dataset
        unique_datasets = sorted(set(filtered_datasets))
        type_data = {}
        
        for model in unique_models:
            type_data[model] = []
            for dataset in unique_datasets:
                indices = [i for i, (m, d) in enumerate(zip(filtered_models, filtered_datasets)) 
                          if m == model and d == dataset]
                if indices:
                    type_data[model].append(filtered_accuracies[indices[0]])
                else:
                    type_data[model].append(0)
        
        # Plotting
        bar_width = 0.25
        index = np.arange(len(unique_datasets))
        
        plt.bar(index - bar_width, type_data['RDLNN'], bar_width, label='RDLNN', color='blue')
        plt.bar(index, type_data['DWT'], bar_width, label='DWT', color='green')
        plt.bar(index + bar_width, type_data['DyWT'], bar_width, label='DyWT', color='red')
        
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')
        plt.title(f'Model Accuracy Comparison - {dataset_type.capitalize()} Images')
        plt.xticks(index, unique_datasets, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'accuracy_comparison_{dataset_type}.png'))
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Main function to run the tests"""
    args = setup_argument_parser()
    
    # Set up the tqdm-compatible logging first
    setup_tqdm_logging()
    
    # Configure GPU if requested
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        
        # Set memory usage monitoring
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        logger.info(f"Initial GPU memory usage: {start_memory/1024**2:.2f} MB")
    else:
        if args.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Using CPU instead.")
        device = torch.device("cpu")
        logger.info("Using CPU for processing")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find test datasets - handle both the dataset directory and authentic/forged subdirectories
    test_datasets = []
    
    # First check if we have a structure with dataset folders containing authentic/forged subfolders
    for item in os.listdir(args.test_dir):
        item_path = os.path.join(args.test_dir, item)
        
        if os.path.isdir(item_path):
            # Check if this directory contains authentic and forged subdirectories
            authentic_dir = os.path.join(item_path, 'authentic')
            forged_dir = os.path.join(item_path, 'forged')
            
            if os.path.isdir(authentic_dir) and os.path.isdir(forged_dir):
                # This is a dataset with authentic/forged subdirectories
                test_datasets.append(authentic_dir)
                test_datasets.append(forged_dir)
                logger.info(f"Found dataset {item} with authentic and forged subdirectories")
            else:
                # This is a standalone directory, add it directly
                test_datasets.append(item_path)
    
    if not test_datasets:
        logger.error(f"No test datasets found in {args.test_dir}")
        return
    
    logger.info(f"Found {len(test_datasets)} test datasets/subdirectories")
    
    # Clear summary file
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("MODEL TESTING SUMMARY\n")
        f.write("=" * 60 + "\n")
    
    # Load and test models
    rdlnn_results = {}
    dwt_results = {}
    dywt_results = {}
    
    # Test DWT model if available
    if DWT_AVAILABLE and args.dwt_model and os.path.exists(args.dwt_model):
        try:
            logger.info(f"Loading DWT model from {args.dwt_model}")
            dwt_model = load_dwt_model(args.dwt_model)
            
            for dataset in test_datasets:
                dataset_result = test_dwt_model(
                    dwt_model, dataset, args.output_dir, 
                    threshold=args.threshold or 0.5, 
                    limit=args.test_limit,
                    device=device
                )
                # Store results by dataset path
                dwt_results[dataset] = dataset_result
                
                # Free GPU memory after each dataset if using GPU
                if args.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error testing DWT model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"DWT model not found or module not available: {args.dwt_model}")
    
    # Test DyWT model if available
    if DYWT_AVAILABLE and args.dywt_model and os.path.exists(args.dywt_model):
        try:
            logger.info(f"Loading DyWT model from {args.dywt_model}")
            dywt_model = load_dywt_model(args.dywt_model)
            
            for dataset in test_datasets:
                dataset_result = test_dywt_model(
                    dywt_model, dataset, args.output_dir, 
                    threshold=args.threshold or 0.5, 
                    limit=args.test_limit,
                    device=device
                )
                # Store results by dataset path
                dywt_results[dataset] = dataset_result
                
                # Free GPU memory after each dataset if using GPU
                if args.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error testing DyWT model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"DyWT model not found or module not available: {args.dywt_model}")

    # Test RDLNN model if available
    if RDLNN_AVAILABLE and args.rdlnn_model and os.path.exists(args.rdlnn_model):
        try:
            logger.info(f"Loading RDLNN model from {args.rdlnn_model}")
            # Load model with specific device
            map_location = device if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
            rdlnn_model = RegressionDLNN.load(args.rdlnn_model)
            
            # Make sure model is on the correct device
            if args.use_gpu and torch.cuda.is_available():
                rdlnn_model.model = rdlnn_model.model.to(device)
                rdlnn_model.device = device
                logger.info(f"RDLNN model loaded and moved to {device}")
            
            for dataset in test_datasets:
                dataset_result = test_rdlnn_model(
                    rdlnn_model, dataset, args.output_dir, 
                    threshold=args.threshold, limit=args.test_limit,
                    device=device
                )
                # Store results by dataset path
                rdlnn_results[dataset] = dataset_result
                
                # Free GPU memory after each dataset
                if args.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error testing RDLNN model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"RDLNN model not found or module not available: {args.rdlnn_model}")
    
    # Create visualizations if requested
    if args.visualize and (rdlnn_results or dwt_results or dywt_results):
        logger.info("Generating result visualizations")
        visualize_results(rdlnn_results, dwt_results, dywt_results, args.output_dir)
    
    # Print final summary
    logger.info("Testing complete! Results saved to {}".format(args.output_dir))
    logger.info("See summary.txt for overall results")
    
    # If using GPU, print final memory usage
    if args.use_gpu and torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        logger.info(f"Final GPU memory usage: {current_memory/1024**2:.2f} MB")
        logger.info(f"Peak GPU memory usage: {peak_memory/1024**2:.2f} MB")
        
        # Report any memory leaks
        if current_memory > start_memory + 10*1024*1024:  # More than 10MB difference
            logger.warning(f"Possible memory leak detected: {(current_memory-start_memory)/1024**2:.2f} MB not released")
        
        # Final cleanup
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()