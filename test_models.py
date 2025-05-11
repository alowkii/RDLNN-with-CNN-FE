#!/usr/bin/env python3
"""
Comprehensive test script for image forgery detection models.
Tests RDLNN, DWT, and DyWT models on the provided test datasets.
Fixed version to handle import issues.
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
    
    return parser.parse_args()

def test_rdlnn_model(model, dataset_dir, output_dir, threshold=None, limit=None):
    """
    Test RDLNN model on a dataset
    
    Args:
        model: Loaded RDLNN model
        dataset_dir: Directory containing test images
        output_dir: Directory to save results
        threshold: Classification threshold (uses model's threshold if None)
        limit: Maximum number of images to test (tests all if None)
    
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
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images (RDLNN)"):
        try:
            start_time = time.time()
            
            # Extract features using the detector
            feature_vector = detector.extract_features(str(img_path))
            
            if feature_vector is None:
                logger.warning(f"Failed to extract features from {img_path}")
                continue
                
            # Reshape to 2D array for prediction
            feature_vector = feature_vector.reshape(1, -1)
            
            # Apply feature selection if available in the model
            if hasattr(model, 'feature_selector') and model.feature_selector is not None:
                try:
                    feature_vector = model.feature_selector(feature_vector)
                except Exception as e:
                    logger.warning(f"Feature selection failed: {e}")
            
            # Get predictions
            predictions, probabilities = model.predict(feature_vector)
            
            # Store results
            results['predictions'].append(predictions[0])
            results['probabilities'].append(probabilities[0])
            results['filenames'].append(os.path.basename(img_path))
            results['processing_times'].append(time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    # Calculate statistics
    results['accuracy'] = calculate_accuracy(results, dataset_dir)
    results['avg_time'] = np.mean(results['processing_times'])
    
    # Save results
    save_results(results, dataset_dir, output_dir, "rdlnn")
    
    return results

def test_dwt_model(model, dataset_dir, output_dir, threshold=0.5, limit=None):
    """
    Test DWT model on a dataset
    
    Args:
        model: Loaded DWT model
        dataset_dir: Directory containing test images
        output_dir: Directory to save results
        threshold: Classification threshold
        limit: Maximum number of images to test
    
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
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images (DWT)"):
        try:
            start_time = time.time()
            
            # Extract DWT features
            features = extract_dwt_features(str(img_path))
            
            if features is None:
                logger.warning(f"Failed to extract features from {img_path}")
                continue
            
            # Make prediction with the model
            prediction = model.predict([features])[0]
            
            # Store results
            results['predictions'].append(prediction)
            results['filenames'].append(os.path.basename(img_path))
            results['processing_times'].append(time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    # Calculate statistics
    results['accuracy'] = calculate_accuracy(results, dataset_dir)
    results['avg_time'] = np.mean(results['processing_times'])
    
    # Save results
    save_results(results, dataset_dir, output_dir, "dwt")
    
    return results

def test_dywt_model(model, dataset_dir, output_dir, threshold=0.5, limit=None):
    """
    Test DyWT model on a dataset
    
    Args:
        model: Loaded DyWT model
        dataset_dir: Directory containing test images
        output_dir: Directory to save results
        threshold: Classification threshold
        limit: Maximum number of images to test
    
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
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images (DyWT)"):
        try:
            start_time = time.time()
            
            # Extract DyWT features
            features = extract_dyadic_wavelet_features(str(img_path))
            
            if features is None:
                logger.warning(f"Failed to extract features from {img_path}")
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
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    # Calculate statistics
    results['accuracy'] = calculate_accuracy(results, dataset_dir)
    results['avg_time'] = np.mean(results['processing_times'])
    
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
    
    # Create a unique filename based on dataset and model
    dataset_name = os.path.basename(dataset_dir)
    output_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_results.csv")
    
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
            if 'probabilities' in results:
                row.append(f"{results['probabilities'][i]:.6f}")
            row.append(f"{results['processing_times'][i]:.6f}")
            
            f.write(','.join(row) + '\n')
    
    logger.info(f"Results saved to {output_file}")
    
    # Also save summary information
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'a') as f:
        f.write(f"\n{model_name.upper()} results for {dataset_name}:\n")
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
        rdlnn_results: Dictionary with RDLNN test results
        dwt_results: Dictionary with DWT test results
        dywt_results: Dictionary with DyWT test results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect accuracy data
    model_names = []
    dataset_names = []
    accuracy_values = []
    
    # RDLNN results
    for dataset_dir, results in rdlnn_results.items():
        model_names.append('RDLNN')
        dataset_names.append(os.path.basename(dataset_dir))
        accuracy_values.append(results['accuracy'])
    
    # DWT results
    for dataset_dir, results in dwt_results.items():
        model_names.append('DWT')
        dataset_names.append(os.path.basename(dataset_dir))
        accuracy_values.append(results['accuracy'])
    
    # DyWT results
    for dataset_dir, results in dywt_results.items():
        model_names.append('DyWT')
        dataset_names.append(os.path.basename(dataset_dir))
        accuracy_values.append(results['accuracy'])
    
    # Create a bar chart for accuracy comparison
    plt.figure(figsize=(12, 8))
    
    # Group by dataset
    unique_datasets = sorted(set(dataset_names))
    unique_models = ['RDLNN', 'DWT', 'DyWT']
    
    # Create data for grouped bar chart
    data = {}
    for model in unique_models:
        data[model] = []
        for dataset in unique_datasets:
            indices = [i for i, (m, d) in enumerate(zip(model_names, dataset_names)) 
                       if m == model and d == dataset]
            if indices:
                data[model].append(accuracy_values[indices[0]])
            else:
                data[model].append(0)  # No data for this dataset
    
    # Plotting
    bar_width = 0.25
    index = np.arange(len(unique_datasets))
    
    # Plot each model's results
    plt.bar(index - bar_width, data['RDLNN'], bar_width, label='RDLNN', color='blue')
    plt.bar(index, data['DWT'], bar_width, label='DWT', color='green')
    plt.bar(index + bar_width, data['DyWT'], bar_width, label='DyWT', color='red')
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison by Dataset')
    plt.xticks(index, unique_datasets, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # Create another plot for processing time comparison
    plt.figure(figsize=(12, 8))
    
    # Collect timing data
    processing_times = {}
    for model in unique_models:
        processing_times[model] = []
        for dataset in unique_datasets:
            if model == 'RDLNN':
                results_dict = rdlnn_results
            elif model == 'DWT':
                results_dict = dwt_results
            else:  # DyWT
                results_dict = dywt_results
                
            # Find matching dataset
            for dir_name, results in results_dict.items():
                if os.path.basename(dir_name) == dataset:
                    processing_times[model].append(results['avg_time'])
                    break
            else:
                processing_times[model].append(0)  # No data for this dataset
    
    # Plotting
    plt.bar(index - bar_width, processing_times['RDLNN'], bar_width, label='RDLNN', color='blue')
    plt.bar(index, processing_times['DWT'], bar_width, label='DWT', color='green')
    plt.bar(index + bar_width, processing_times['DyWT'], bar_width, label='DyWT', color='red')
    
    plt.xlabel('Dataset')
    plt.ylabel('Average Processing Time (s)')
    plt.title('Model Processing Time Comparison by Dataset')
    plt.xticks(index, unique_datasets, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'processing_time_comparison.png'))
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Main function to run the tests"""
    args = setup_argument_parser()
    
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
    
    # Test RDLNN model if available
    if RDLNN_AVAILABLE and args.rdlnn_model and os.path.exists(args.rdlnn_model):
        try:
            logger.info(f"Loading RDLNN model from {args.rdlnn_model}")
            rdlnn_model = RegressionDLNN.load(args.rdlnn_model)
            
            for dataset in test_datasets:
                rdlnn_results[dataset] = test_rdlnn_model(
                    rdlnn_model, dataset, args.output_dir, 
                    threshold=args.threshold, limit=args.test_limit
                )
        except Exception as e:
            logger.error(f"Error testing RDLNN model: {e}")
    else:
        logger.warning(f"RDLNN model not found or module not available: {args.rdlnn_model}")
    
    # Test DWT model if available
    if DWT_AVAILABLE and args.dwt_model and os.path.exists(args.dwt_model):
        try:
            logger.info(f"Loading DWT model from {args.dwt_model}")
            dwt_model = load_dwt_model(args.dwt_model)
            
            for dataset in test_datasets:
                dwt_results[dataset] = test_dwt_model(
                    dwt_model, dataset, args.output_dir, 
                    threshold=args.threshold or 0.5, limit=args.test_limit
                )
        except Exception as e:
            logger.error(f"Error testing DWT model: {e}")
    else:
        logger.warning(f"DWT model not found or module not available: {args.dwt_model}")
    
    # Test DyWT model if available
    if DYWT_AVAILABLE and args.dywt_model and os.path.exists(args.dywt_model):
        try:
            logger.info(f"Loading DyWT model from {args.dywt_model}")
            dywt_model = load_dywt_model(args.dywt_model)
            
            for dataset in test_datasets:
                dywt_results[dataset] = test_dywt_model(
                    dywt_model, dataset, args.output_dir, 
                    threshold=args.threshold or 0.5, limit=args.test_limit
                )
        except Exception as e:
            logger.error(f"Error testing DyWT model: {e}")
    else:
        logger.warning(f"DyWT model not found or module not available: {args.dywt_model}")
    
    # Create visualizations if requested
    if args.visualize and (rdlnn_results or dwt_results or dywt_results):
        logger.info("Generating result visualizations")
        visualize_results(rdlnn_results, dwt_results, dywt_results, args.output_dir)
    
    # Print final summary
    logger.info("Testing complete! Results saved to {}".format(args.output_dir))
    logger.info("See summary.txt for overall results")

if __name__ == "__main__":
    main()