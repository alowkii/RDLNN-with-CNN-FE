#!/usr/bin/env python3
"""
API Server for Image Forgery Detection System
Provides REST endpoints to access model predictions, training results, and processed images
"""

import os
import sys
import json
import logging
import time
import argparse
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
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
    'detection_model_path': 'data/models/forgery_detection_model.pth',
    'localization_model_path': 'data/models/pdywt_localizer.pth',
    'upload_folder': 'uploads',
    'results_folder': 'data/results',
    'threshold': 0.7,  # Default classification threshold
    'batch_size': 8,
    'log_file': 'data/results/api_server.log'
}

# Set up logging
def configure_logging():
    """Configure logging for the API server"""
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

@app.route('/api/run/detection', methods=['POST'])
def run_detection():
    """Run the main.py detection directly with parameters"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Get threshold from request or use default
        threshold = request.form.get('threshold', config['threshold'])
        if threshold is None:
            threshold = 0.7
        
        # Create a unique output directory for this run
        run_id = f"run_{int(time.time())}"
        output_dir = os.path.join(config['results_folder'], run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare command
        cmd = [
            sys.executable,
            'main.py',
            '--mode', 'single',
            '--image_path', filepath,
            '--model_path', config['detection_model_path'],
            '--output_dir', output_dir,
            '--threshold', str(threshold)
        ]
        
        # Run the command
        import subprocess
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output
        stdout = process.stdout
        stderr = process.stderr
        
        # Read result file
        result_filename = f"{os.path.splitext(filename)[0]}_result.txt"
        result_path = os.path.join(output_dir, result_filename)
        
        result = {}
        if os.path.exists(result_path):
            result = parse_detection_result(result_path)
        
        # Build response
        response = {
            'run_id': run_id,
            'command': ' '.join(cmd),
            'status': 'success' if process.returncode == 0 else 'error',
            'return_code': process.returncode,
            'result': result,
            'output_dir': output_dir,
            'stdout': stdout,
            'stderr': stderr
        }
        
        # Return JSON response
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running detection: {e}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/run/localization', methods=['POST'])
def run_localization():
    """Run the main.py localization directly with parameters"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Get threshold from request or use default
        threshold = request.form.get('threshold', config['threshold'])
        if threshold is None:
            threshold = 0.7
        
        # Create a unique output directory for this run
        run_id = f"run_{int(time.time())}"
        output_dir = os.path.join(config['results_folder'], run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare command
        cmd = [
            sys.executable,
            'main.py',
            '--mode', 'localize',
            '--image_path', filepath,
            '--model_path', config['detection_model_path'],
            '--localization_model_path', config['localization_model_path'],
            '--output_dir', output_dir,
            '--threshold', str(threshold)
        ]
        
        # Run the command
        import subprocess
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output
        stdout = process.stdout
        stderr = process.stderr
        
        # Find forgery map
        base_name = os.path.splitext(filename)[0]
        forgery_map = f"{base_name}_forgery_map.png"
        forgery_map_path = os.path.join(output_dir, forgery_map)
        
        # Build response
        response = {
            'run_id': run_id,
            'command': ' '.join(cmd),
            'status': 'success' if process.returncode == 0 else 'error',
            'return_code': process.returncode,
            'output_dir': output_dir,
            'stdout': stdout,
            'stderr': stderr
        }
        
        # Add forgery map if it exists
        if os.path.exists(forgery_map_path):
            relative_path = os.path.relpath(forgery_map_path, config['results_folder'])
            response['forgery_map'] = f"/results/{relative_path}"
            
        # Get result info if available
        result_filename = f"{base_name}_result.txt"
        result_path = os.path.join(output_dir, result_filename)
        
        if os.path.exists(result_path):
            response['result'] = parse_detection_result(result_path)
        
        # Return JSON response
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running localization: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize the models and processors
def initialize_models():
    """Initialize the detection and localization models"""
    global detection_model, detector, batch_processor
    
    logger.info("Initializing models and processors...")
    
    try:
        # Create upload and results directories if they don't exist
        os.makedirs(config['upload_folder'], exist_ok=True)
        os.makedirs(config['results_folder'], exist_ok=True)
        
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
        return False

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API server is running"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': detection_model is not None or detector is not None
    })
    
@app.route('/api/test_data/info', methods=['GET'])
def get_test_data_info():
    """Get information about available test data and results"""
    try:
        results_info = {}
        
        # Check for test_image.jpg
        test_image_path = os.path.join(config['results_folder'], 'test_image.jpg')
        if os.path.exists(test_image_path):
            results_info['test_image'] = {
                'path': '/results/test_image.jpg',
                'size': os.path.getsize(test_image_path),
                'modified': os.path.getmtime(test_image_path)
            }
        
        # Check for test_image_result.txt
        result_path = os.path.join(config['results_folder'], 'test_image_result.txt')
        if os.path.exists(result_path):
            results_info['detection_result'] = {
                'path': '/results/test_image_result.txt',
                'size': os.path.getsize(result_path),
                'modified': os.path.getmtime(result_path),
                'content': parse_detection_result(result_path)
            }
        
        # Check for forgery map
        forgery_map_path = os.path.join(config['results_folder'], 'test_image_forgery_map.png')
        if os.path.exists(forgery_map_path):
            results_info['forgery_map'] = {
                'path': '/results/test_image_forgery_map.png',
                'size': os.path.getsize(forgery_map_path),
                'modified': os.path.getmtime(forgery_map_path)
            }
        
        # Check for logs
        logs = {}
        for log_file in ['forgery_detection.log', 'api_server.log']:
            log_path = os.path.join(config['results_folder'], log_file)
            if os.path.exists(log_path):
                logs[log_file] = {
                    'path': f'/results/{log_file}',
                    'size': os.path.getsize(log_path),
                    'modified': os.path.getmtime(log_path)
                }
        
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
        
        return jsonify({
            'available_test_data': len(results_info) > 0,
            'results': results_info
        })
        
    except Exception as e:
        logger.error(f"Error getting test data info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Return the current configuration"""
    # Filter out sensitive information
    safe_config = {
        'threshold': config['threshold'],
        'batch_size': config['batch_size'],
        'models': {
            'detection': os.path.basename(config['detection_model_path']),
            'localization': os.path.basename(config['localization_model_path'])
        },
        'results_folder': config['results_folder']
    }
    return jsonify(safe_config)

@app.route('/api/detect', methods=['POST'])
def detect_forgery():
    """Detect if an image is forged"""
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
        
        # Process the image
        start_time = time.time()
        result = detector.detect(filepath)
        
        # Format the result
        response = {
            'filename': filename,
            'result': 'forged' if result['prediction'] == 1 else 'authentic',
            'probability': float(result['probability']),
            'threshold': float(detector.threshold),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time())
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in forgery detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/localize', methods=['POST'])
def localize_forgery():
    """Detect and localize forgery in an image"""
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
        
        # First detect if image is forged
        start_time = time.time()
        detection_result = detector.detect(filepath)
        
        # Determine result type
        is_forged = detection_result['prediction'] == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Base response
        response = {
            'filename': filename,
            'result': result_type,
            'probability': float(detection_result['probability']),
            'threshold': float(detector.threshold),
            'processing_time': 0,
            'timestamp': int(time.time())
        }
        
        # If forged, perform localization
        if is_forged:
            # Generate a unique filename for the forgery map
            basename = os.path.splitext(filename)[0]
            forgery_map_filename = f"{basename}_forgery_map.png"
            forgery_map_path = os.path.join(config['results_folder'], forgery_map_filename)
            
            # Localize the forgery
            localization_result = detector.localize(filepath, save_path=forgery_map_path)
            
            # Update the response with localization information
            response['forgery_map'] = f"/results/{forgery_map_filename}"
            response['regions'] = localization_result.get('region_proposals', [])
        
        # Update processing time
        response['processing_time'] = time.time() - start_time
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in forgery localization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch/detect', methods=['POST'])
def batch_detect():
    """Process multiple images in batch mode"""
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'images[]' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    try:
        # Get all uploaded files
        files = request.files.getlist('images[]')
        
        if not files:
            return jsonify({'error': 'No valid files provided'}), 400
        
        # Create a temporary directory for the batch
        batch_id = f"batch_{int(time.time())}"
        batch_dir = os.path.join(config['upload_folder'], batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save all files
        filepaths = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(batch_dir, filename)
            file.save(filepath)
            filepaths.append(filepath)
        
        # Process batch
        start_time = time.time()
        batch_results = []
        
        # Process each image individually
        for filepath in filepaths:
            try:
                filename = os.path.basename(filepath)
                result = detector.detect(filepath)
                
                batch_results.append({
                    'filename': filename,
                    'result': 'forged' if result['prediction'] == 1 else 'authentic',
                    'probability': float(result['probability'])
                })
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                batch_results.append({
                    'filename': os.path.basename(filepath),
                    'error': str(e)
                })
        
        # Create response
        response = {
            'batch_id': batch_id,
            'total': len(files),
            'successful': len([r for r in batch_results if 'error' not in r]),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'batch_results': batch_results
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<path:filename>', methods=['GET'])
def get_result_file(filename):
    """Serve result files (e.g., forgery maps)"""
    try:
        filepath = os.path.join(config['results_folder'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath)
        
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/results/image/<filename>', methods=['GET'])
def get_result_image(filename):
    """Get a specific result image"""
    try:
        # Look for the image file
        filepath = os.path.join(config['results_folder'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(filepath)
        
    except Exception as e:
        logger.error(f"Error serving result image {filename}: {e}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/results/log/<filename>', methods=['GET'])
def get_result_log(filename):
    """Get a specific log file content"""
    try:
        # Look for the log file
        filepath = os.path.join(config['results_folder'], filename)
        
        if not os.path.exists(filepath) or not filename.endswith('.log'):
            return jsonify({'error': 'Log file not found'}), 404
        
        # Read the log file
        with open(filepath, 'r') as f:
            content = f.readlines()
        
        return jsonify({
            'filename': filename,
            'lines': content,
            'size': os.path.getsize(filepath),
            'modified': os.path.getmtime(filepath)
        })
        
    except Exception as e:
        logger.error(f"Error serving log file {filename}: {e}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/results/detection/<filename>', methods=['GET'])
def get_detection_result(filename):
    """Get a specific detection result file content"""
    try:
        # Look for the result file
        filepath = os.path.join(config['results_folder'], filename)
        
        if not os.path.exists(filepath) or not filename.endswith('_result.txt'):
            return jsonify({'error': 'Detection result file not found'}), 404
        
        # Parse the result file
        result = parse_detection_result(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error serving detection result {filename}: {e}")
        return jsonify({'error': str(e)}), 500
        
def parse_detection_result(filepath):
    """Parse a detection result text file into a structured format"""
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
            if key in ['confidence', 'threshold']:
                try:
                    value = float(value)
                except:
                    pass
            
            result[key] = value
        
        # Add the filename
        result['filename'] = os.path.basename(filepath)
        
        # Find related images
        base_name = os.path.splitext(os.path.basename(filepath))[0].replace('_result', '')
        
        # Look for forgery map
        forgery_map = f"{base_name}_forgery_map.png"
        forgery_map_path = os.path.join(os.path.dirname(filepath), forgery_map)
        
        if os.path.exists(forgery_map_path):
            result['forgery_map'] = f"/results/{forgery_map}"
        
        # Look for original image
        image_extensions = ['.jpg', '.jpeg', '.png']
        for ext in image_extensions:
            image_path = os.path.join(os.path.dirname(filepath), f"{base_name}{ext}")
            if os.path.exists(image_path):
                result['image'] = f"/results/{base_name}{ext}"
                break
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing detection result: {e}")
        return {'error': str(e)}

@app.route('/api/results/list', methods=['GET'])
def list_results():
    """List available result files"""
    try:
        # Get all files in the results directory
        result_files = []
        for root, dirs, files in os.walk(config['results_folder']):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.txt', '.log')):
                    rel_path = os.path.relpath(os.path.join(root, file), config['results_folder'])
                    file_path = os.path.join(root, file)
                    result_files.append({
                        'filename': file,
                        'path': f"/results/{rel_path}",
                        'size': os.path.getsize(file_path),
                        'modified': os.path.getmtime(file_path),
                        'type': get_file_type(file)
                    })
        
        return jsonify({
            'total': len(result_files),
            'results': result_files
        })
        
    except Exception as e:
        logger.error(f"Error listing results: {e}")
        return jsonify({'error': str(e)}), 500
        
def get_file_type(filename):
    """Determine file type based on filename"""
    if filename.endswith('.log'):
        return 'log'
    elif filename.endswith('_forgery_map.png'):
        return 'forgery_map'
    elif filename.endswith('_result.txt'):
        return 'detection_result'
    elif filename.endswith('.png'):
        return 'image'
    elif filename.endswith(('.jpg', '.jpeg')):
        return 'image'
    else:
        return 'other'

@app.route('/api/threshold/<float:value>', methods=['POST'])
def set_threshold(value):
    """Set the detection threshold"""
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    try:
        # Validate the threshold value
        if value < 0 or value > 1:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        # Set the threshold
        detector.threshold = value
        config['threshold'] = value
        
        return jsonify({
            'success': True,
            'message': f'Threshold set to {value}',
            'threshold': value
        })
        
    except Exception as e:
        logger.error(f"Error setting threshold: {e}")
        return jsonify({'error': str(e)}), 500

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='API Server for Image Forgery Detection')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on')
    
    parser.add_argument('--port', type=int, default=5000,
                      help='Port to run the server on')
    
    parser.add_argument('--detection-model', type=str,
                      default='data/models/forgery_detection_model.pth',
                      help='Path to the detection model')
    
    parser.add_argument('--localization-model', type=str,
                      default='data/models/pdywt_localizer.pth',
                      help='Path to the localization model')
    
    parser.add_argument('--threshold', type=float, default=None,
                      help='Override classification threshold')
    
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for processing')
    
    parser.add_argument('--upload-folder', type=str, default='uploads',
                      help='Folder to store uploaded files')
    
    parser.add_argument('--results-folder', type=str, default='data/results',
                      help='Folder to store results')
    
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode')
    
    return parser.parse_args()

@app.route('/api/reproduce', methods=['GET'])
def reproduce_commands():
    """Get the commands used to produce the test data"""
    commands = [
        {
            "type": "detection",
            "command": "python main.py --mode single --image_path .\\data\\test_image.jpg --model_path .\\data\\models\\pdywt_model.pth --output_dir data/results --threshold 0.7",
            "description": "Detect forgery in test_image.jpg"
        },
        {
            "type": "localization",
            "command": "python main.py --mode localize --image_path .\\data\\test_image.jpg --model_path .\\data\\models\\pdywt_model.pth --localization_model_path .\\data\\models\\pdywt_localizer.pth --output_dir data/results --threshold 0.7",
            "description": "Localize forgery in test_image.jpg"
        }
    ]
    
    return jsonify({
        "commands": commands,
        "note": "These commands were used to produce the test data in data/results/"
    })

@app.route('/api/tail_log/<filename>', methods=['GET'])
def tail_log(filename):
    """Get the last N lines of a log file"""
    try:
        # Limit to log files only
        if not filename.endswith('.log'):
            return jsonify({'error': 'Only log files are supported'}), 400
            
        # Get lines parameter (default: 50)
        lines = request.args.get('lines', 50, type=int)
        if lines <= 0:
            lines = 50
            
        # Get the log file path
        log_path = os.path.join(config['results_folder'], filename)
        if not os.path.exists(log_path):
            return jsonify({'error': f'Log file {filename} not found'}), 404
            
        # Read the last N lines
        with open(log_path, 'r') as f:
            content = f.readlines()
            
        # Get the last N lines
        last_lines = content[-lines:] if len(content) > lines else content
        
        return jsonify({
            'filename': filename,
            'total_lines': len(content),
            'lines_returned': len(last_lines),
            'content': last_lines
        })
        
    except Exception as e:
        logger.error(f"Error tailing log file {filename}: {e}")
        return jsonify({'error': str(e)}), 500

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