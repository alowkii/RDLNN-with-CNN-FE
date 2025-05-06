import os
import sys
import json
import logging
import time
import re
import argparse
import traceback
import numpy as np
import torch
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, TiffTags
import io
import base64
from datetime import datetime
import mimetypes

if '.tiff' not in mimetypes.types_map:
    mimetypes.add_type('image/tiff', '.tiff')
if '.tif' not in mimetypes.types_map:
    mimetypes.add_type('image/tiff', '.tif')

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
    'localization_model_path': '../data/models/cnn_localizer.pth',
    'upload_folder': './api/uploads',  # Temporary upload location
    'threshold': 0.675,
    'batch_size': 32,
    'log_file': './api_server.log'
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

# Ensure upload directory exists
def ensure_directories():
    """
    Ensure only necessary directories exist
    Creates only the temporary upload directory
    """
    # Create upload directory
    os.makedirs(config['upload_folder'], exist_ok=True)
    
    logger.info(f"Upload directory verified and ready")

# Convert image to base64 string
def image_to_base64(image_path, format='JPEG'):
    """
    Convert an image file to a base64 encoded string
    
    Args:
        image_path (str): Path to the image file
        format (str): Image format to use for encoding
    
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

# Convert PIL Image object to base64 string
def pil_image_to_base64(pil_image, format='PNG'):
    """
    Convert a PIL Image object to a base64 encoded string
    
    Args:
        pil_image (PIL.Image): PIL Image object
        format (str): Image format to use for encoding
    
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        logger.error(f"Error converting PIL Image to base64: {e}")
        return None

# Format a dictionary as a text string
def format_dict_to_text(data):
    """
    Format a dictionary as a text string with one key-value pair per line
    
    Args:
        data (dict): Dictionary to format
    
    Returns:
        str: Formatted text string
    """
    lines = []
    # Write standard headers first
    standard_keys = [
        'image', 'result', 'probability', 'threshold', 
        'processing_time', 'timestamp', 'model'
    ]
    
    for key in standard_keys:
        if key in data:
            lines.append(f"{key.replace('_', ' ').title()}: {data[key]}")
    
    # Write any remaining data
    for key, value in data.items():
        if key not in standard_keys:
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(lines)

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
        'threshold': config['threshold']
    })
    
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
        }
    }
    return jsonify(safe_config)

@app.route('/api/detect', methods=['POST'])
def detect_forgery():
    """
    Detect if an image is forged
    Uses the configured detection model and returns forgery probability
    Returns results in JSON with base64 encoded image
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily to process with the detector
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)

        # Add logging for file type
        import mimetypes
        file_type = mimetypes.guess_type(filepath)[0]
        logger.info(f"Processing file of type: {file_type}")

        # Get more detailed image information
        try:
            with Image.open(filepath) as img:
                format_info = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                }
                # For TIFF images, gather additional metadata
                if img.format == 'TIFF':
                    tiff_tags = {}
                    for tag, value in img.tag.items():
                        tag_name = TiffTags.TAGS.get(tag, tag)
                        tiff_tags[tag_name] = value
                    format_info['tiff_tags'] = tiff_tags
                logger.info(f"Processing file: {filename}, type: {file_type}, info: {format_info}")
        except Exception as e:
            logger.warning(f"Failed to get detailed image information: {e}")
            logger.info(f"Processing file of type: {file_type}")
        
        # Process the image
        start_time = time.time()
        result = detector.detect(filepath)
        
        # Check if there's an error in the result
        if 'error' in result:
            return jsonify({
                'error': result['error'],
                'filename': filename
            }), 500
        
        # Handle missing prediction key
        if 'prediction' not in result:
            logger.warning(f"Missing 'prediction' key in detection result: {result}")
            # Use probability if available, otherwise default to 0
            probability = result.get('probability', 0)
            prediction = 1 if probability >= detector.threshold else 0
        else:
            # Use existing prediction
            prediction = result['prediction']
            probability = result.get('probability', 0.5)
        
        # Determine result type
        is_forged = prediction == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Get the original image as base64
        original_image_base64 = image_to_base64(filepath)
        
        # Format the result data
        result_data = {
            'image': filename,
            'result': result_type,
            'probability': f"{float(probability):.4f}",
            'threshold': f"{float(detector.threshold):.4f}",
            'processing_time': f"{time.time() - start_time:.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': os.path.basename(config['detection_model_path'])
        }
        
        # Format result text
        result_text = format_dict_to_text(result_data)
        
        # Create response with the base64 encoded image
        run_id = f"run_{int(time.time())}"
        response = {
            'run_id': run_id,
            'filename': filename,
            'result': result_type,
            'probability': float(probability),
            'threshold': float(detector.threshold),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'original_image_base64': original_image_base64,
            'result_text': result_text
        }
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")
        
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
    Returns results in JSON with base64 encoded images
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
        
        # First detect if image is forged
        start_time = time.time()
        detection_result = detector.detect(filepath)
        
        # Determine result type
        is_forged = detection_result['prediction'] == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Get the original image as base64
        original_image_base64 = image_to_base64(filepath)
        
        # Base result data
        result_data = {
            'image': filename,
            'result': result_type,
            'probability': f"{float(detection_result['probability']):.4f}",
            'threshold': f"{float(detector.threshold):.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': os.path.basename(config['detection_model_path'])
        }
        
        # Base response
        run_id = f"run_{int(time.time())}"
        response = {
            'run_id': run_id,
            'filename': filename,
            'result': result_type,
            'probability': float(detection_result['probability']),
            'threshold': float(detector.threshold),
            'processing_time': 0,
            'timestamp': int(time.time()),
            'original_image_base64': original_image_base64
        }
        
        # If forged, perform localization and add forgery map
        if is_forged:
            # Create a temporary file path for the forgery map
            forgery_map_path = os.path.join(config['upload_folder'], f"temp_forgery_map_{run_id}.png")
            
            # Localize the forgery
            localization_result = detector.localize(filepath, save_path=forgery_map_path)
            
            # Convert forgery map to base64
            forgery_map_base64 = image_to_base64(forgery_map_path, format='PNG')
            
            # Update result data with localization info
            result_data['localization_model'] = os.path.basename(config['localization_model_path'])
            result_data['region_count'] = len(localization_result.get('region_proposals', []))
            
            # Update the response with localization information
            response['forgery_map_base64'] = forgery_map_base64
            response['regions'] = localization_result.get('region_proposals', [])
            
            # Remove temporary forgery map file
            try:
                os.remove(forgery_map_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary forgery map {forgery_map_path}: {e}")
        
        # Update processing time
        processing_time = time.time() - start_time
        result_data['processing_time'] = f"{processing_time:.4f}"
        response['processing_time'] = processing_time
        
        # Format result text
        response['result_text'] = format_dict_to_text(result_data)
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")
        
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
    Returns results in JSON with base64 encoded images
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
        
        # Process batch
        start_time = time.time()
        batch_results = []
        batch_summary_lines = [
            f"Batch ID: {batch_id}",
            f"Processed: {datetime.now().isoformat()}",
            f"Total images: {len(files)}",
            f"Detection threshold: {detector.threshold}",
            "\n--- RESULTS ---\n"
        ]
        
        # Process each image individually
        for file in files:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(config['upload_folder'], filename)
                file.save(filepath)
                
                # Process the image
                result = detector.detect(filepath)
                is_forged = result['prediction'] == 1
                result_type = 'forged' if is_forged else 'authentic'
                
                # Get the original image as base64
                original_image_base64 = image_to_base64(filepath)
                
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
                
                # Format result text
                result_text = format_dict_to_text(result_data)
                
                # Append to batch summary
                batch_summary_lines.append(f"Image: {filename}")
                batch_summary_lines.append(f"Result: {result_type}")
                batch_summary_lines.append(f"Probability: {result_data['probability']}\n")
                
                # Create result object for response
                image_result = {
                    'filename': filename,
                    'result': result_type,
                    'probability': float(result['probability']),
                    'original_image_base64': original_image_base64,
                    'result_text': result_text
                }
                
                batch_results.append(image_result)
                
                # Clean up the temporary file
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {filepath}: {e}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                logger.error(traceback.format_exc())
                
                # Log the error to batch summary
                batch_summary_lines.append(f"Image: {filename}")
                batch_summary_lines.append(f"Error: {str(e)}\n")
                
                batch_results.append({
                    'filename': filename,
                    'error': str(e)
                })
        
        # Finalize batch summary
        processing_time = time.time() - start_time
        batch_summary_lines.append(f"\nTotal processing time: {processing_time:.4f} seconds")
        batch_summary_lines.append(f"Average per image: {processing_time / len(files):.4f} seconds")
        batch_summary_text = "\n".join(batch_summary_lines)
        
        # Create response
        response = {
            'batch_id': batch_id,
            'total': len(files),
            'successful': len([r for r in batch_results if 'error' not in r]),
            'processing_time': processing_time,
            'timestamp': int(time.time()),
            'batch_results': batch_results,
            'batch_summary_text': batch_summary_text
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
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
                'upload': upload_dir_info
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

def parse_args():
    """Parse command line arguments for the API server"""
    parser = argparse.ArgumentParser(description='Stateless Image Forgery Detection API Server')
    
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
    parser.add_argument('--localization-model', type=str, default='../data/models/cnn_localizer.pth',
                        help='Path to localization model')
    
    # Directories
    parser.add_argument('--upload-folder', type=str, default='./uploads',
                        help='Temporary folder for uploaded images')
    
    # Processing parameters
    parser.add_argument('--threshold', type=float, default=0.675,
                        help='Detection threshold')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing')
    
    return parser.parse_args()

# Record the server startup time
startup_time = time.time()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Update configuration
    config.update({
        'detection_model_path': args.detection_model,
        'localization_model_path': args.localization_model,
        'threshold': args.threshold,
        'batch_size': args.batch_size,
        'upload_folder': args.upload_folder
    })
    
    # Configure logging
    logger = configure_logging()
    
    # Create necessary directories
    os.makedirs(config['upload_folder'], exist_ok=True)
    
    # Log API server startup
    logger.info("-" * 80)
    logger.info("Starting Image Forgery Detection API Server")
    logger.info(f"Detection model: {config['detection_model_path']}")
    logger.info(f"Localization model: {config['localization_model_path']}")
    
    # Initialize models
    if initialize_models():
        # Start the server
        logger.info(f"Starting API server on {args.host}:{args.port}...")
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        logger.error("Failed to initialize models. Exiting...")
        sys.exit(1)