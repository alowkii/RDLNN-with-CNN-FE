#!/usr/bin/env python3
"""
API Server for Image Forgery Detection System
Provides REST endpoints to access model predictions, training results, and processed images

Features:
- Consistent file path management across all endpoints
- Directory creation and validation
- Detailed error handling with traceback information
- Comprehensive logging
- Standardized result format and structure
"""

import os
import sys
import json
import logging
import time
import re
import argparse
import time  # Ensure time is imported

# Record the server startup time
startup_time = time.time()
import traceback
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import shutil
from datetime import datetime

# Import modules from the forgery detection system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.rdlnn import RegressionDLNN
from modules.feature_extractor import PDyWTCNNDetector
from modules.utils import setup_logging, logger
from modules.batch_processor import OptimizedBatchProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables for models and processors
detection_model = None
detector = None
batch_processor = None

# Configuration
config = {
    'detection_model_path': '../data/models/rdlnn_model.pth',
    'localization_model_path': '../data/models/pdywt_localizer.pth',
    'upload_folder': './api/uploads',
    'results_folder': '../data/results',
    'threshold': 0.7,
    'batch_size': 8,
    'log_file': '../data/results/api_server.log'
}

# Set up logging
def configure_logging():
    """
    Configure logging for the API server
    Sets up both file and console handlers
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
    
    # Set up file handler
    file_handler = logging.FileHandler(config['log_file'])
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Ensure all required directories exist
def ensure_directories():
    """
    Ensure all required directories exist
    Creates parent directories and standard subdirectories for results
    """
    # Create primary directories
    os.makedirs(config['upload_folder'], exist_ok=True)
    os.makedirs(config['results_folder'], exist_ok=True)
    
    # Create standard subdirectories for results organization
    subdirs = ['detection', 'localization', 'logs', 'batch_results', 'test_data']
    for subdir in subdirs:
        os.makedirs(os.path.join(config['results_folder'], subdir), exist_ok=True)
    
    logger.info(f"Directory structure verified and ready")

# Unified path management for result files
def get_result_paths(filename, run_id=None, result_type='detection'):
    """
    Get standardized paths for result files
    
    Args:
        filename (str): Original image filename
        run_id (str, optional): Specific run ID, or creates a new one if None
        result_type (str, optional): Type of result ('detection' or 'localization')
    
    Returns:
        dict: Dictionary containing all relevant paths and URLs
    """
    # Get base filename without extension
    basename = os.path.splitext(filename)[0]
    
    # Create or use run_id
    if not run_id:
        run_id = f"run_{int(time.time())}"
    
    # Use result type as subdirectory to organize results
    result_subdir = result_type.lower()
    if result_subdir not in ['detection', 'localization', 'batch_results']:
        result_subdir = 'detection'  # Default if not recognized
    
    # Create full output directory path
    output_dir = os.path.join(config['results_folder'], result_subdir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create standard paths and URLs
    original_path = os.path.join(output_dir, filename)
    result_path = os.path.join(output_dir, f"{basename}_result.txt")
    forgery_map_path = os.path.join(output_dir, f"{basename}_forgery_map.png")
    
    # URLs use forward slashes for browser compatibility
    base_url = f"/results/{result_subdir}/{run_id}"
    
    return {
        'run_id': run_id,
        'result_type': result_type,
        'output_dir': output_dir,
        'original_path': original_path,
        'result_path': result_path,
        'forgery_map_path': forgery_map_path,
        'original_url': f"{base_url}/{filename}",
        'result_url': f"{base_url}/{basename}_result.txt",
        'forgery_map_url': f"{base_url}/{basename}_forgery_map.png"
    }

# Save result information to a standardized text file
def save_result_info(result_path, data):
    """
    Save result information to a text file in a standardized format
    
    Args:
        result_path (str): Path to save the result file
        data (dict): Dictionary containing result data
    """
    try:
        with open(result_path, 'w') as f:
            # Write standard headers first
            standard_keys = [
                'image', 'result', 'probability', 'threshold', 
                'processing_time', 'timestamp', 'model'
            ]
            
            for key in standard_keys:
                if key in data:
                    f.write(f"{key.replace('_', ' ').title()}: {data[key]}\n")
            
            # Write any remaining data
            for key, value in data.items():
                if key not in standard_keys:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        logger.debug(f"Result info saved to: {result_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving result info: {e}")
        return False

# Parse a detection result text file into a structured format
def parse_detection_result(filepath):
    """
    Parse a detection result text file into a structured format
    
    Args:
        filepath (str): Path to the result file
    
    Returns:
        dict: Dictionary containing parsed result data
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        result = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
                
            key = parts[0].strip().lower().replace(' ', '_')
            value = parts[1].strip()
            
            # Try to convert numerical values
            if key in ['confidence', 'threshold', 'probability', 'processing_time']:
                try:
                    value = float(value)
                except:
                    pass
            
        result['filename'] = os.path.basename(filepath)
        
        # Find related files in the same directory
        dir_path = os.path.dirname(filepath)
        
        # Extract base name (remove _result.txt)
        base_name = os.path.splitext(os.path.basename(filepath))[0].replace('_result', '')
        
        # Look for forgery map
        forgery_map = f"{base_name}_forgery_map.png"
        forgery_map_path = os.path.join(dir_path, forgery_map)
        
        if os.path.exists(forgery_map_path):
            # Convert to relative URL
            rel_path = os.path.relpath(forgery_map_path, config['results_folder'])
            result['forgery_map'] = f"/results/{rel_path.replace(os.sep, '/')}"
        
        # Look for original image
        image_extensions = ['.jpg', '.jpeg', '.png']
        for ext in image_extensions:
            image_path = os.path.join(dir_path, f"{base_name}{ext}")
            if os.path.exists(image_path):
                rel_path = os.path.relpath(image_path, config['results_folder'])
                result['image'] = f"/results/{rel_path.replace(os.sep, '/')}"
                break
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing detection result: {e}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}

# Initialize the models and processors
def initialize_models():
    """
    Initialize the detection and localization models
    Creates required directories and loads all models
    
    Returns:
        bool: True if models initialized successfully, False otherwise
    """
    global detection_model, detector, batch_processor
    
    logger.info("Initializing models and processors...")
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Initialize the detector (includes both detection and localization)
        detector = PDyWTCNNDetector(
            model_path=config['detection_model_path'],
            localization_model_path=config['localization_model_path']
        )
        
        # If a specific threshold was provided, override the default
        if config['threshold'] is not None:
            detector.threshold = config['threshold']
        
        # Initialize batch processor if we have a detection model
        if detector:
            batch_processor = OptimizedBatchProcessor(batch_size=config['batch_size'])
        
        logger.info("Models initialized successfully")
        logger.info(f"Using detection threshold: {detector.threshold:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        logger.error(traceback.format_exc())
        return False

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Endpoint to check if the API server is running
    Returns server status, timestamp, and model status
    """
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': detection_model is not None or detector is not None,
        'upload_folder': os.path.abspath(config['upload_folder']),
        'results_folder': os.path.abspath(config['results_folder']),
        'threshold': config['threshold']
    })
    
@app.route('/api/test_data/info', methods=['GET'])
def get_test_data_info():
    """
    Get information about available test data and results
    Checks standard test data locations and returns metadata
    """
    try:
        results_info = {}
        
        # Create test data directory if it doesn't exist
        test_data_dir = os.path.join(config['results_folder'], 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Check for test_image.jpg in multiple locations
        potential_test_images = [
            os.path.join(config['results_folder'], 'test_image.jpg'),
            os.path.join(test_data_dir, 'test_image.jpg')
        ]
        
        for test_path in potential_test_images:
            if os.path.exists(test_path):
                rel_path = os.path.relpath(test_path, config['results_folder'])
                results_info['test_image'] = {
                    'path': f"/results/{rel_path.replace(os.sep, '/')}",
                    'size': os.path.getsize(test_path),
                    'modified': os.path.getmtime(test_path),
                    'abs_path': os.path.abspath(test_path)
                }
                break
        
        # Check for test_image_result.txt in multiple locations
        for base_dir in [config['results_folder'], test_data_dir]:
            result_path = os.path.join(base_dir, 'test_image_result.txt')
            if os.path.exists(result_path):
                rel_path = os.path.relpath(result_path, config['results_folder'])
                results_info['detection_result'] = {
                    'path': f"/results/{rel_path.replace(os.sep, '/')}",
                    'size': os.path.getsize(result_path),
                    'modified': os.path.getmtime(result_path),
                    'content': parse_detection_result(result_path)
                }
                break
        
        # Check for forgery map in multiple locations
        for base_dir in [config['results_folder'], test_data_dir]:
            forgery_map_path = os.path.join(base_dir, 'test_image_forgery_map.png')
            if os.path.exists(forgery_map_path):
                rel_path = os.path.relpath(forgery_map_path, config['results_folder'])
                results_info['forgery_map'] = {
                    'path': f"/results/{rel_path.replace(os.sep, '/')}",
                    'size': os.path.getsize(forgery_map_path),
                    'modified': os.path.getmtime(forgery_map_path)
                }
                break
        
        # Check for logs
        logs = {}
        for log_file in ['forgery_detection.log', 'api_server.log']:
            for base_dir in [config['results_folder'], os.path.join(config['results_folder'], 'logs')]:
                log_path = os.path.join(base_dir, log_file)
                if os.path.exists(log_path):
                    rel_path = os.path.relpath(log_path, config['results_folder'])
                    logs[log_file] = {
                        'path': f"/results/{rel_path.replace(os.sep, '/')}",
                        'size': os.path.getsize(log_path),
                        'modified': os.path.getmtime(log_path)
                    }
                    break
        
        if logs:
            results_info['logs'] = logs
        
        # Check for threshold_analysis.png
        threshold_path = os.path.join(config['results_folder'], 'threshold_analysis.png')
        if os.path.exists(threshold_path):
            results_info['threshold_analysis'] = {
                'path': '/results/threshold_analysis.png',
                'size': os.path.getsize(threshold_path),
                'modified': os.path.getmtime(threshold_path)
            }
        
        # Check if we need to create test data
        if not results_info:
            results_info['message'] = "No test data found. Use /api/save_test_image to create test data."
        
        return jsonify({
            'available_test_data': len(results_info) > 0,
            'results': results_info
        })
        
    except Exception as e:
        logger.error(f"Error getting test data info: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/save_test_image', methods=['POST'])
def save_test_image():
    """
    Save an uploaded image as test_image.jpg in the results folder
    This provides a standard test image for development and testing
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Create test data directory
        test_data_dir = os.path.join(config['results_folder'], 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Save the file to both locations for maximum compatibility
        filename = secure_filename(file.filename)
        
        # Save to main results directory
        main_test_path = os.path.join(config['results_folder'], 'test_image.jpg')
        
        # Save to test_data subdirectory
        test_data_path = os.path.join(test_data_dir, 'test_image.jpg')
        
        # Open and save with PIL to ensure format compatibility
        img = Image.open(file)
        img.save(main_test_path)
        img.save(test_data_path)
        
        logger.info(f"Test image saved to {main_test_path} and {test_data_path}")
        
        # For clarity, also save with the original filename
        original_name_path = os.path.join(test_data_dir, filename)
        img.save(original_name_path)
        
        return jsonify({
            'success': True,
            'message': 'Test image saved successfully',
            'original_filename': filename,
            'test_image_paths': [
                '/results/test_image.jpg',
                f'/results/test_data/test_image.jpg',
                f'/results/test_data/{filename}'
            ]
        })
        
    except Exception as e:
        logger.error(f"Error saving test image: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Return the current configuration
    Filters out sensitive information for security
    """
    # Create a safe version of the config without sensitive information
    safe_config = {
        'threshold': config['threshold'],
        'batch_size': config['batch_size'],
        'models': {
            'rdlnn': os.path.basename(config['detection_model_path']),
            'localization': os.path.basename(config['localization_model_path'])
        },
        'results_folder': config['results_folder']
    }
    return jsonify(safe_config)

@app.route('/api/detect', methods=['POST'])
def detect_forgery():
    """
    Detect if an image is forged
    Uses the configured detection model and returns forgery probability
    Saves results in a consistent format and location
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Create paths for results using standardized function
        paths = get_result_paths(filename, result_type='detection')
        
        # Save a copy of the original image to the results directory
        shutil.copy2(filepath, paths['original_path'])
        
        # Process the image
        start_time = time.time()
        result = detector.detect(filepath)
        
        # Determine result type
        is_forged = result['prediction'] == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Format the result data for saving
        result_data = {
            'image': filename,
            'result': result_type,
            'probability': f"{float(result['probability']):.4f}",
            'threshold': f"{float(detector.threshold):.4f}",
            'processing_time': f"{time.time() - start_time:.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': os.path.basename(config['detection_model_path'])
        }
        
        # Save result info to text file
        save_result_info(paths['result_path'], result_data)
        
        # Format the response
        response = {
            'run_id': paths['run_id'],
            'filename': filename,
            'result': result_type,
            'probability': float(result['probability']),
            'threshold': float(detector.threshold),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'original_image': paths['original_url'],
            'result_file': paths['result_url']
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in forgery detection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/api/localize', methods=['POST'])
def localize_forgery():
    """
    Detect and localize forgery in an image
    First detects if the image is forged, then if positive, localizes the forgery regions
    Saves all results and generates a forgery map image
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Create paths for results using standardized function
        paths = get_result_paths(filename, result_type='localization')
        
        # Save a copy of the original image to the results directory
        shutil.copy2(filepath, paths['original_path'])
        
        # First detect if image is forged
        start_time = time.time()
        detection_result = detector.detect(filepath)
        
        # Determine result type
        is_forged = detection_result['prediction'] == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Base result data for saving
        result_data = {
            'image': filename,
            'result': result_type,
            'probability': f"{float(detection_result['probability']):.4f}",
            'threshold': f"{float(detector.threshold):.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': os.path.basename(config['detection_model_path'])
        }
        
        # Base response
        response = {
            'run_id': paths['run_id'],
            'filename': filename,
            'result': result_type,
            'probability': float(detection_result['probability']),
            'threshold': float(detector.threshold),
            'processing_time': 0,
            'timestamp': int(time.time()),
            'original_image': paths['original_url'],
            'result_file': paths['result_url']
        }
        
        # If forged, perform localization
        if is_forged:
            # Localize the forgery
            localization_result = detector.localize(filepath, save_path=paths['forgery_map_path'])
            
            # Update result data with localization info
            result_data['localization_model'] = os.path.basename(config['localization_model_path'])
            result_data['region_count'] = len(localization_result.get('region_proposals', []))
            
            # Update the response with localization information
            response['forgery_map'] = paths['forgery_map_url']
            response['regions'] = localization_result.get('region_proposals', [])
        
        # Update processing time
        processing_time = time.time() - start_time
        result_data['processing_time'] = f"{processing_time:.4f}"
        response['processing_time'] = processing_time
        
        # Save result info to text file
        save_result_info(paths['result_path'], result_data)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in forgery localization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/api/batch/detect', methods=['POST'])
def batch_detect():
    """
    Process multiple images in batch mode
    Handles multiple uploaded images and processes them efficiently
    Saves results in a dedicated batch results directory
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'images[]' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    try:
        # Get all uploaded files
        files = request.files.getlist('images[]')
        
        if not files:
            return jsonify({'error': 'No valid files provided'}), 400
        
        # Create a unique batch ID
        batch_id = f"batch_{int(time.time())}"
        
        # Create a dedicated directory for batch results
        batch_results_dir = os.path.join(config['results_folder'], 'batch_results', batch_id)
        os.makedirs(batch_results_dir, exist_ok=True)
        
        # Create a temporary directory for uploads
        batch_uploads_dir = os.path.join(config['upload_folder'], batch_id)
        os.makedirs(batch_uploads_dir, exist_ok=True)
        
        # Save all files
        filepaths = []
        filenames = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(batch_uploads_dir, filename)
            file.save(filepath)
            filepaths.append(filepath)
            filenames.append(filename)
        
        # Create a batch summary file
        summary_path = os.path.join(batch_results_dir, 'batch_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Batch ID: {batch_id}\n")
            f.write(f"Processed: {datetime.now().isoformat()}\n")
            f.write(f"Total images: {len(files)}\n")
            f.write(f"Image files: {', '.join(filenames)}\n")
            f.write(f"Detection threshold: {detector.threshold}\n")
            f.write("\n--- RESULTS ---\n\n")
        
        # Process batch
        start_time = time.time()
        batch_results = []
        
        # Process each image individually and save results
        for filepath in filepaths:
            try:
                filename = os.path.basename(filepath)
                
                # Create paths for this image's results
                image_paths = get_result_paths(filename, run_id=batch_id, result_type='batch_results')
                
                # Save a copy of the original image to results
                shutil.copy2(filepath, image_paths['original_path'])
                
                # Process the image
                result = detector.detect(filepath)
                is_forged = result['prediction'] == 1
                result_type = 'forged' if is_forged else 'authentic'
                
                # Save result information
                result_data = {
                    'image': filename,
                    'result': result_type,
                    'probability': f"{float(result['probability']):.4f}",
                    'threshold': f"{float(detector.threshold):.4f}",
                    'timestamp': datetime.now().isoformat(),
                    'model': os.path.basename(config['detection_model_path']),
                    'batch_id': batch_id
                }
                
                # Save result to file
                save_result_info(image_paths['result_path'], result_data)
                
                # Append result to batch summary
                with open(summary_path, 'a') as f:
                    f.write(f"Image: {filename}\n")
                    f.write(f"Result: {result_type}\n")
                    f.write(f"Probability: {result_data['probability']}\n\n")
                
                # Create result object for response
                image_result = {
                    'filename': filename,
                    'result': result_type,
                    'probability': float(result['probability']),
                    'original_image': image_paths['original_url'],
                    'result_file': image_paths['result_url']
                }
                
                batch_results.append(image_result)
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                logger.error(traceback.format_exc())
                
                # Log the error to batch summary
                with open(summary_path, 'a') as f:
                    f.write(f"Image: {os.path.basename(filepath)}\n")
                    f.write(f"Error: {str(e)}\n\n")
                
                batch_results.append({
                    'filename': os.path.basename(filepath),
                    'error': str(e)
                })
        
        # Finalize batch summary
        processing_time = time.time() - start_time
        with open(summary_path, 'a') as f:
            f.write(f"\nTotal processing time: {processing_time:.4f} seconds\n")
            f.write(f"Average per image: {processing_time / len(files):.4f} seconds\n")
        
        # Create response
        response = {
            'batch_id': batch_id,
            'total': len(files),
            'successful': len([r for r in batch_results if 'error' not in r]),
            'processing_time': processing_time,
            'timestamp': int(time.time()),
            'batch_results': batch_results,
            'summary_file': f"/results/batch_results/{batch_id}/batch_summary.txt"
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
# Helper functions for result file management

def get_file_type(filename):
    """
    Determine file type based on filename
    Provides detailed categorization of different file types
    """
    if filename.endswith('.log'):
        return 'log'
    elif filename.endswith('_forgery_map.png'):
        return 'forgery_map'
    elif filename.endswith('_result.txt'):
        return 'detection_result'
    elif filename.endswith('batch_summary.txt'):
        return 'batch_summary'
    elif filename.endswith('.png'):
        return 'image'
    elif filename.endswith(('.jpg', '.jpeg')):
        return 'image'
    elif filename.endswith('.txt'):
        return 'text'
    elif filename.endswith('.json'):
        return 'json'
    elif filename.endswith(('.pth', '.pt')):
        return 'model'
    else:
        return 'other'

def get_result_category(rel_path, file_type):
    """
    Categorize results based on path and file type
    
    Args:
        rel_path (str): Relative path from results folder
        file_type (str): File type from get_file_type
        
    Returns:
        str: Category name
    """
    path_parts = rel_path.split(os.sep)
    
    # Check standard subdirectories
    if len(path_parts) > 0:
        if path_parts[0] in ['detection', 'localization', 'batch_results', 'test_data', 'logs']:
            return path_parts[0]
    
    # Check for run directories
    for part in path_parts:
        if part.startswith('run_'):
            return 'run_results'
        elif part.startswith('batch_'):
            return 'batch_results'
    
    # Categorize by file type
    if file_type == 'log':
        return 'logs'
    elif file_type == 'model':
        return 'models'
    elif file_type in ['forgery_map', 'detection_result']:
        return 'detection_results'
    elif file_type == 'image':
        return 'images'
    
    return 'other'

def format_file_size(size_bytes):
    """
    Format file size in human-readable format
    
    Args:
        size_bytes (int): File size in bytes
        
    Returns:
        str: Formatted size string (e.g., "2.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def count_file_types(result_files):
    """
    Count occurrences of each file type
    
    Args:
        result_files (list): List of result file objects
        
    Returns:
        dict: Count of each file type
    """
    counts = {}
    for result in result_files:
        file_type = result['type']
        if file_type not in counts:
            counts[file_type] = 0
        counts[file_type] += 1
    return counts

def find_related_files(result_files):
    """
    Group related files (e.g., image, forgery map, and result file)
    
    Args:
        result_files (list): List of result file objects
        
    Returns:
        dict: Groups of related files
    """
    groups = {}
    
    # First, organize files by basename (without extension)
    basename_map = {}
    for result in result_files:
        filename = result['filename']
        
        # Extract base name
        base = os.path.splitext(filename)[0]
        
        # Remove common suffixes
        for suffix in ['_result', '_forgery_map']:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        
        # Group by directory + base name
        dir_base = f"{result['directory']}:{base}"
        if dir_base not in basename_map:
            basename_map[dir_base] = []
        basename_map[dir_base].append(result)
    
    # Keep only groups with multiple files
    related_groups = []
    for base, files in basename_map.items():
        if len(files) > 1:
            # Create a group with organized structure
            group = {
                'base_name': base.split(':')[1],
                'directory': base.split(':')[0],
                'files': {},
                'count': len(files)
            }
            
            # Organize by file type
            for file in files:
                file_type = file['type']
                if file_type not in group['files']:
                    group['files'][file_type] = []
                group['files'][file_type].append(file)
                
            related_groups.append(group)
    
    return related_groups

# Check if files need to be copied to standard test locations
def create_test_data():
    """
    Create default test data if requested
    Copies a sample image to test_image.jpg location
    """
    logger.info("Creating default test data...")
    test_data_dir = os.path.join(config['results_folder'], 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Check if any image files exist in the upload folder
    image_found = False
    sample_image = None
    
    for file in os.listdir(config['upload_folder']):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(config['upload_folder'], file)):
            sample_image = os.path.join(config['upload_folder'], file)
            image_found = True
            break
            
    # If no image found in uploads, look in the project directory
    if not image_found:
        for directory in ['data', '.', 'test', 'sample']:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(directory, file)):
                        sample_image = os.path.join(directory, file)
                        image_found = True
                        break
                if image_found:
                    break
    
    if image_found:
        logger.info(f"Using {sample_image} as sample test image")
        
        # Copy to standard test locations
        main_test_path = os.path.join(config['results_folder'], 'test_image.jpg')
        test_data_path = os.path.join(test_data_dir, 'test_image.jpg')
        
        try:
            # Use PIL to ensure format compatibility
            img = Image.open(sample_image)
            img.save(main_test_path)
            img.save(test_data_path)
            logger.info(f"Created test image at {main_test_path} and {test_data_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating test data: {e}")
            return False
    else:
        logger.warning("No sample images found to create test data")
        return False

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    """
    Get information about available models
    Returns metadata about detection and localization models
    """
    try:
        models_info = {}
        
        # Check detection model
        detection_model_path = config['detection_model_path']
        if os.path.exists(detection_model_path):
            models_info['detection_model'] = {
                'path': detection_model_path,
                'filename': os.path.basename(detection_model_path),
                'size': os.path.getsize(detection_model_path),
                'modified': os.path.getmtime(detection_model_path),
                'threshold': config['threshold']
            }
        
        # Check RDLNN model path
        detection_model_path = config.get('detection_model_path')
        if detection_model_path and os.path.exists(detection_model_path):
            models_info['rdlnn_model'] = {
                'path': detection_model_path,
                'filename': os.path.basename(detection_model_path),
                'size': os.path.getsize(detection_model_path),
                'modified': os.path.getmtime(detection_model_path)
            }
        
        # Check localization model
        localization_model_path = config['localization_model_path']
        if os.path.exists(localization_model_path):
            models_info['localization_model'] = {
                'path': localization_model_path,
                'filename': os.path.basename(localization_model_path),
                'size': os.path.getsize(localization_model_path),
                'modified': os.path.getmtime(localization_model_path)
            }
        
        # Current active model
        if detector:
            model_type = "Unknown"
            if hasattr(detector, 'rdlnn_model') and detector.rdlnn_model:
                model_type = "RegressionDLNN"
            elif hasattr(detector, 'detection_model') and detector.detection_model:
                model_type = "WaveletCNN"
            elif hasattr(detector, 'flat_feature_detector') and detector.flat_feature_detector:
                model_type = "FlatFeatureDetector"
            
            models_info['active_model'] = {
                'type': model_type,
                'threshold': detector.threshold
            }
        
        return jsonify(models_info)
        
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """
    Get detailed system status information
    Returns server and model status, directories, and environment info
    """
    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        cuda_info = None
        if cuda_available:
            cuda_info = {
                'device_name': torch.cuda.get_device_name(0),
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated(0),
                'memory_reserved': torch.cuda.memory_reserved(0)
            }
        
        # Get directory info
        upload_dir_info = {
            'path': os.path.abspath(config['upload_folder']),
            'exists': os.path.isdir(config['upload_folder']),
            'writable': os.access(config['upload_folder'], os.W_OK) if os.path.isdir(config['upload_folder']) else False,
            'file_count': len(os.listdir(config['upload_folder'])) if os.path.isdir(config['upload_folder']) else 0
        }
        
        results_dir_info = {
            'path': os.path.abspath(config['results_folder']),
            'exists': os.path.isdir(config['results_folder']),
            'writable': os.access(config['results_folder'], os.W_OK) if os.path.isdir(config['results_folder']) else False,
            'subdirs': next(os.walk(config['results_folder']))[1] if os.path.isdir(config['results_folder']) else []
        }
        
        # Check model status
        model_status = {
            'detection_model_loaded': detector is not None,
            'threshold': config['threshold'],
            'batch_processor': batch_processor is not None
        }
        
        # Combined status
        status_info = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - startup_time,
            'environment': {
                'python_version': sys.version,
                'pytorch_version': torch.__version__,
                'cuda_available': cuda_available,
                'cuda_info': cuda_info
            },
            'directories': {
                'upload': upload_dir_info,
                'results': results_dir_info
            },
            'models': model_status,
            'config': {
                'batch_size': config['batch_size'],
                'threshold': config['threshold']
            }
        }
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/api/results/<path:path>', methods=['GET'])
def serve_result_file(path):
    """
    Serve a result file from the results directory
    
    Args:
        path: Relative path to the file within the results folder
    """
    file_path = os.path.join(config['results_folder'], path)
    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {path}'}), 404
    
    try:
        return send_file(file_path)
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/detection/<run_id>', methods=['GET'])
def get_detection_result(run_id):
    """
    Retrieve detection result for a specific run
    
    Args:
        run_id: ID of the detection run
        
    Returns:
        Dictionary with detection results including original image, 
        result data and file paths
    """
    try:
        # Build path to the run directory
        run_dir = os.path.join(config['results_folder'], 'detection', run_id)
        
        if not os.path.exists(run_dir) or not os.path.isdir(run_dir):
            return jsonify({'error': f'Detection run not found: {run_id}'}), 404
        
        # Get all files in the run directory
        files = os.listdir(run_dir)
        
        # Find result file, original image, and any other related files
        result_file = None
        original_image = None
        
        # Get first image and its associated result file
        for file in files:
            if file.endswith('_result.txt'):
                result_file = file
                # Base name without _result.txt
                base_name = file[:-11]
                
                # Look for original image with this base name
                for img_file in files:
                    if img_file.startswith(base_name) and img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        original_image = img_file
                        break
                
                if original_image:
                    break
        
        if not result_file or not original_image:
            return jsonify({'error': 'No complete detection result found in this run'}), 404
        
        # Get full paths
        result_path = os.path.join(run_dir, result_file)
        image_path = os.path.join(run_dir, original_image)
        
        # Read result file content
        result_content = {}
        with open(result_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    result_content[key.strip().lower().replace(' ', '_')] = value.strip()
        
        # Create response with file URLs
        response = {
            'run_id': run_id,
            'original_image': f"/results/detection/{run_id}/{original_image}",
            'result_file': f"/results/detection/{run_id}/{result_file}",
            'result_data': result_content,
            'timestamp': int(time.time())
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error retrieving detection result: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/results/localization/<run_id>', methods=['GET'])
def get_localization_result(run_id):
    """
    Retrieve localization result for a specific run
    
    Args:
        run_id: ID of the localization run
        
    Returns:
        Dictionary with localization results including original image, 
        forgery map, result data and file paths
    """
    try:
        # Build path to the run directory
        run_dir = os.path.join(config['results_folder'], 'localization', run_id)
        
        if not os.path.exists(run_dir) or not os.path.isdir(run_dir):
            return jsonify({'error': f'Localization run not found: {run_id}'}), 404
        
        # Get all files in the run directory
        files = os.listdir(run_dir)
        
        # Find result file, original image, and forgery map
        result_file = None
        original_image = None
        forgery_map = None
        
        # Get first result set
        for file in files:
            if file.endswith('_result.txt'):
                result_file = file
                # Base name without _result.txt
                base_name = file[:-11]
                
                # Look for original image and forgery map with this base name
                for related_file in files:
                    if related_file.startswith(base_name):
                        if related_file.endswith('_forgery_map.png'):
                            forgery_map = related_file
                        elif related_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not related_file.endswith('_forgery_map.png'):
                            original_image = related_file
                
                if original_image:
                    break
        
        if not result_file or not original_image:
            return jsonify({'error': 'No complete localization result found in this run'}), 404
        
        # Get full paths
        result_path = os.path.join(run_dir, result_file)
        image_path = os.path.join(run_dir, original_image)
        
        # Read result file content
        result_content = {}
        with open(result_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    result_content[key.strip().lower().replace(' ', '_')] = value.strip()
        
        # Create response with file URLs
        response = {
            'run_id': run_id,
            'original_image': f"/results/localization/{run_id}/{original_image}",
            'result_file': f"/results/localization/{run_id}/{result_file}",
            'result_data': result_content,
            'timestamp': int(time.time())
        }
        
        # Add forgery map if available
        if forgery_map:
            response['forgery_map'] = f"/results/localization/{run_id}/{forgery_map}"
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error retrieving localization result: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/results/latest', methods=['GET'])
def get_latest_results():
    """
    Retrieve the latest detection and localization results
    
    Returns:
        Dictionary with the most recent results from each category
    """
    try:
        result_types = ['detection', 'localization']
        latest_results = {}
        
        for result_type in result_types:
            type_dir = os.path.join(config['results_folder'], result_type)
            if not os.path.exists(type_dir):
                continue
                
            # Get all run directories
            run_dirs = [d for d in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, d))]
            
            if not run_dirs:
                continue
                
            # Sort by timestamp (format: run_TIMESTAMP)
            run_dirs.sort(key=lambda x: int(x.split('_')[-1]) if x.startswith('run_') and x.split('_')[-1].isdigit() else 0, reverse=True)
            
            # Get the latest run
            latest_run = run_dirs[0]
            
            # Get result for this run using the appropriate endpoint
            if result_type == 'detection':
                result = get_detection_result(latest_run).get_json()
            elif result_type == 'localization':
                result = get_localization_result(latest_run).get_json()
            else:
                continue
                
            # Add to results if no error
            if 'error' not in result:
                latest_results[result_type] = result
        
        if not latest_results:
            return jsonify({'message': 'No results found'}), 404
            
        return jsonify(latest_results)
        
    except Exception as e:
        logger.error(f"Error retrieving latest results: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/detect/image', methods=['POST'])
def detect_single_image():
    """
    Process a single image for forgery detection
    Ensures only one image is processed and result is returned directly
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Create a unique run ID for this specific detection
        run_id = f"run_{int(time.time())}"
        
        # Create output directory
        output_dir = os.path.join(config['results_folder'], 'detection', run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save a copy of the original image
        original_path = os.path.join(output_dir, filename)
        shutil.copy2(filepath, original_path)
        
        # Process the image
        start_time = time.time()
        result = detector.detect(filepath)
        
        # Determine result type
        is_forged = result['prediction'] == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Create result file path
        base_name = os.path.splitext(filename)[0]
        result_path = os.path.join(output_dir, f"{base_name}_result.txt")
        
        # Format the result data for saving
        result_data = {
            'image': filename,
            'result': result_type,
            'probability': f"{float(result['probability']):.4f}",
            'threshold': f"{float(detector.threshold):.4f}",
            'processing_time': f"{time.time() - start_time:.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': os.path.basename(config['detection_model_path'])
        }
        
        # Save result info to text file
        with open(result_path, 'w') as f:
            for key, value in result_data.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        # Construct URLs for frontend
        original_url = f"/results/detection/{run_id}/{filename}"
        result_url = f"/results/detection/{run_id}/{base_name}_result.txt"
        
        # Format the response
        response = {
            'run_id': run_id,
            'filename': filename,
            'result': result_type,
            'probability': float(result['probability']),
            'threshold': float(detector.threshold),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'original_image': original_url,
            'result_file': result_url
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in single image detection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/localize/image', methods=['POST'])
def localize_single_image():
    """
    Process a single image for forgery localization
    Ensures only one image is processed and result is returned directly
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Create a unique run ID for this specific localization
        run_id = f"run_{int(time.time())}"
        
        # Create output directory
        output_dir = os.path.join(config['results_folder'], 'localization', run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save a copy of the original image
        original_path = os.path.join(output_dir, filename)
        shutil.copy2(filepath, original_path)
        
        # Get base name and paths for results
        base_name = os.path.splitext(filename)[0]
        result_path = os.path.join(output_dir, f"{base_name}_result.txt")
        forgery_map_path = os.path.join(output_dir, f"{base_name}_forgery_map.png")
        
        # First detect if the image is forged
        start_time = time.time()
        detection_result = detector.detect(filepath)
        
        # Determine result type
        is_forged = detection_result['prediction'] == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Base result data for saving
        result_data = {
            'image': filename,
            'result': result_type,
            'probability': f"{float(detection_result['probability']):.4f}",
            'threshold': f"{float(detector.threshold):.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': os.path.basename(config['detection_model_path'])
        }
        
        # Construct response URLs
        original_url = f"/results/localization/{run_id}/{filename}"
        result_url = f"/results/localization/{run_id}/{base_name}_result.txt"
        
        # Base response
        response = {
            'run_id': run_id,
            'filename': filename,
            'result': result_type,
            'probability': float(detection_result['probability']),
            'threshold': float(detector.threshold),
            'timestamp': int(time.time()),
            'original_image': original_url,
            'result_file': result_url
        }
        
        # If forged, perform localization
        if is_forged:
            # Localize the forgery
            localization_result = detector.localize(filepath, save_path=forgery_map_path)
            
            # Update result data with localization info
            result_data['localization_model'] = os.path.basename(config['localization_model_path'])
            result_data['region_count'] = len(localization_result.get('region_proposals', []))
            
            # Update the response with localization information
            forgery_map_url = f"/results/localization/{run_id}/{base_name}_forgery_map.png"
            response['forgery_map'] = forgery_map_url
            response['regions'] = localization_result.get('region_proposals', [])
        
        # Update processing time
        processing_time = time.time() - start_time
        result_data['processing_time'] = f"{processing_time:.4f}"
        response['processing_time'] = processing_time
        
        # Save result info to text file
        with open(result_path, 'w') as f:
            for key, value in result_data.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in single image localization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/results/list/detection', methods=['GET'])
def list_detection_results():
    """
    List all available detection results ordered by newest first
    """
    try:
        detection_dir = os.path.join(config['results_folder'], 'detection')
        if not os.path.exists(detection_dir):
            return jsonify({'results': []}), 200
            
        # Get all run directories
        run_dirs = [d for d in os.listdir(detection_dir) if os.path.isdir(os.path.join(detection_dir, d))]
        
        # Sort by timestamp (format: run_TIMESTAMP)
        run_dirs.sort(key=lambda x: int(x.split('_')[-1]) if x.startswith('run_') and x.split('_')[-1].isdigit() else 0, reverse=True)
        
        # Process each run directory
        results = []
        for run_id in run_dirs:
            run_dir = os.path.join(detection_dir, run_id)
            
            # Get the first set of detection results in this directory
            files = os.listdir(run_dir)
            result_files = [f for f in files if f.endswith('_result.txt')]
            
            if not result_files:
                continue
                
            # Just take the first result file
            result_file = result_files[0]
            base_name = result_file[:-11]  # Remove '_result.txt'
            
            # Find the original image
            original_image = None
            for f in files:
                if f.startswith(base_name) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not f.endswith('_forgery_map.png'):
                    original_image = f
                    break
            
            if not original_image:
                continue
                
            # Read basic info from result file
            result_path = os.path.join(run_dir, result_file)
            result_content = {}
            with open(result_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result_content[key.strip().lower().replace(' ', '_')] = value.strip()
            
            # Add to results
            result_info = {
                'run_id': run_id,
                'timestamp': int(run_id.split('_')[-1]) if run_id.startswith('run_') and run_id.split('_')[-1].isdigit() else 0,
                'filename': original_image,
                'original_image': f"/results/detection/{run_id}/{original_image}",
                'result_file': f"/results/detection/{run_id}/{result_file}",
                'result': result_content.get('result', 'unknown')
            }
            
            if 'probability' in result_content:
                try:
                    result_info['probability'] = float(result_content['probability'])
                except ValueError:
                    result_info['probability'] = result_content['probability']
            
            results.append(result_info)
            
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error listing detection results: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/results/list/localization', methods=['GET'])
def list_localization_results():
    """
    List all available localization results ordered by newest first
    """
    try:
        localization_dir = os.path.join(config['results_folder'], 'localization')
        if not os.path.exists(localization_dir):
            return jsonify({'results': []}), 200
            
        # Get all run directories
        run_dirs = [d for d in os.listdir(localization_dir) if os.path.isdir(os.path.join(localization_dir, d))]
        
        # Sort by timestamp (format: run_TIMESTAMP)
        run_dirs.sort(key=lambda x: int(x.split('_')[-1]) if x.startswith('run_') and x.split('_')[-1].isdigit() else 0, reverse=True)
        
        # Process each run directory
        results = []
        for run_id in run_dirs:
            run_dir = os.path.join(localization_dir, run_id)
            
            # Get the first set of localization results in this directory
            files = os.listdir(run_dir)
            result_files = [f for f in files if f.endswith('_result.txt')]
            
            if not result_files:
                continue
                
            # Just take the first result file
            result_file = result_files[0]
            base_name = result_file[:-11]  # Remove '_result.txt'
            
            # Find the original image and forgery map
            original_image = None
            forgery_map = None
            
            for f in files:
                if f.startswith(base_name):
                    if f.endswith('_forgery_map.png'):
                        forgery_map = f
                    elif f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not f.endswith('_forgery_map.png'):
                        original_image = f
            
            if not original_image:
                continue
                
            # Read basic info from result file
            result_path = os.path.join(run_dir, result_file)
            result_content = {}
            with open(result_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result_content[key.strip().lower().replace(' ', '_')] = value.strip()
            
            # Add to results
            result_info = {
                'run_id': run_id,
                'timestamp': int(run_id.split('_')[-1]) if run_id.startswith('run_') and run_id.split('_')[-1].isdigit() else 0,
                'filename': original_image,
                'original_image': f"/results/localization/{run_id}/{original_image}",
                'result_file': f"/results/localization/{run_id}/{result_file}",
                'result': result_content.get('result', 'unknown')
            }
            
            if forgery_map:
                result_info['forgery_map'] = f"/results/localization/{run_id}/{forgery_map}"
            
            if 'probability' in result_content:
                try:
                    result_info['probability'] = float(result_content['probability'])
                except ValueError:
                    result_info['probability'] = result_content['probability']
                    
            if 'region_count' in result_content:
                try:
                    result_info['region_count'] = int(result_content['region_count'])
                except ValueError:
                    result_info['region_count'] = result_content['region_count']
            
            results.append(result_info)
            
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error listing localization results: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def parse_args():
    """Parse command line arguments for the API server"""
    parser = argparse.ArgumentParser(description='Image Forgery Detection API Server')
    
    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run server in debug mode')
    
    # Model paths
    parser.add_argument('--detection-model', type=str, default='../data/models/rdlnn_model.pth',
                        help='Path to detection model')
    parser.add_argument('--localization-model', type=str, default='../data/models/pdywt_localizer.pth',
                        help='Path to localization model')
    
    # Directories
    parser.add_argument('--upload-folder', type=str, default='./uploads',
                        help='Folder for uploaded images')
    parser.add_argument('--results-folder', type=str, default='../data/results',
                        help='Folder for results')
    
    # Processing parameters
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Detection threshold')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Update configuration
    config.update({
        'detection_model_path': args.detection_model,
        'localization_model_path': args.localization_model,
        'threshold': args.threshold,
        'batch_size': args.batch_size,
        'upload_folder': args.upload_folder,
        'results_folder': args.results_folder
    })
    
    # Configure logging
    logger = configure_logging()
    
    # Create necessary directories
    os.makedirs(config['upload_folder'], exist_ok=True)
    os.makedirs(config['results_folder'], exist_ok=True)
    
    # Log API server startup
    logger.info("-" * 80)
    logger.info("Starting Image Forgery Detection API Server")
    logger.info(f"Detection model: {config['detection_model_path']}")
    logger.info(f"Localization model: {config['localization_model_path']}")
    logger.info(f"Results folder: {config['results_folder']}")
    
    # Initialize models
    if initialize_models():
        # Start the server
        logger.info(f"Starting API server on {args.host}:{args.port}...")
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        logger.error("Failed to initialize models. Exiting...")
        sys.exit(1)