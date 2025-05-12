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

# Import DWT and DyWT modules
from dwt.utils import load_model as dwt_load_model
from dwt.test import detect_forgery as dwt_detect_forgery
from dywt.utils import load_model as dywt_load_model 
from dywt.test import detect_forgery as dywt_detect_forgery

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables for models and processors
detection_model = None
detector = None
batch_processor = None
dwt_model = None
dywt_model = None

# Configuration
config = {
    'detection_model_path': './data/models/rdlnn_model.pth',
    'localization_model_path': './data/models/localizer.pth',
    'dwt_model_path': './dwt/model/dwt_model.pkl',
    'dywt_model_path': './dywt/model/dywt_model.pkl',
    'upload_folder': './api/uploads',
    'threshold': 0.55,
    'batch_size': 32,
    'log_file': './api/api_server.log'
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
    global detection_model, detector, batch_processor, dwt_model, dywt_model
    
    logger.info("Initializing models and processors...")
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Explicitly import RegressionDLNN to ensure it's available
        from modules.rdlnn import RegressionDLNN
        
        # Log the model paths we're using
        logger.info(f"Using detection model path: {config['detection_model_path']}")
        logger.info(f"Using localization model path: {config['localization_model_path']}")
        logger.info(f"Using DWT model path: {config['dwt_model_path']}")
        logger.info(f"Using DyWT model path: {config['dywt_model_path']}")
        
        # Check if the RDLNN model file exists
        if not os.path.exists(config['detection_model_path']):
            logger.error(f"RDLNN Model file not found: {config['detection_model_path']}")
            return False
            
        # Initialize the detector with explicit model loading
        try:
            # First try loading using direct RDLNN loading
            rdlnn_model = RegressionDLNN.load(config['detection_model_path'])
            logger.info(f"Successfully loaded RDLNN model directly")
            
            # Now initialize detector
            detector = PDyWTCNNDetector(
                model_path=config['detection_model_path'],
                localization_model_path=config['localization_model_path']
            )
            
            # Make sure detector has the RDLNN model
            if not hasattr(detector, 'rdlnn_model') or detector.rdlnn_model is None:
                logger.warning("PDyWTCNNDetector didn't load the RDLNN model; setting it explicitly")
                detector.rdlnn_model = rdlnn_model
                
            # Verify the RDLNN model is correctly loaded
            if hasattr(detector, 'rdlnn_model') and detector.rdlnn_model is not None:
                logger.info("RDLNN model is correctly loaded in detector")
                
                # Check for feature selector
                has_feature_selector = hasattr(detector.rdlnn_model, 'feature_selector') and detector.rdlnn_model.feature_selector is not None
                logger.info(f"RDLNN model has feature selector: {has_feature_selector}")
                
                # Get the model's threshold
                model_threshold = getattr(detector.rdlnn_model, 'threshold', None)
                logger.info(f"RDLNN model threshold: {model_threshold}")
                
                # Set detector threshold to match model if available
                if model_threshold is not None:
                    detector.threshold = model_threshold
                    logger.info(f"Set detector threshold to match model: {detector.threshold}")
            else:
                logger.error("Failed to load RDLNN model in detector after explicit attempt")
                return False
                
        except Exception as rdlnn_error:
            logger.error(f"Error during explicit model loading: {rdlnn_error}")
            logger.error(traceback.format_exc())
            return False
        
        # Initialize DWT model
        if os.path.exists(config['dwt_model_path']):
            try:
                dwt_model = dwt_load_model(config['dwt_model_path'])
                logger.info(f"Successfully loaded DWT model from {config['dwt_model_path']}")
            except Exception as e:
                logger.error(f"Error loading DWT model: {e}")
                logger.error(traceback.format_exc())
                # We don't return False here because we want to continue with other models
        else:
            logger.warning(f"DWT model file not found: {config['dwt_model_path']}")
        
        # Initialize DyWT model
        if os.path.exists(config['dywt_model_path']):
            try:
                dywt_model = dywt_load_model(config['dywt_model_path'])
                logger.info(f"Successfully loaded DyWT model from {config['dywt_model_path']}")
            except Exception as e:
                logger.error(f"Error loading DyWT model: {e}")
                logger.error(traceback.format_exc())
                # We don't return False here because we want to continue with other models
        else:
            logger.warning(f"DyWT model file not found: {config['dywt_model_path']}")
        
        # If a specific threshold was provided in config, override the default
        if config['threshold'] is not None:
            logger.info(f"Overriding threshold with config value: {config['threshold']}")
            detector.threshold = config['threshold']
        
        # Initialize batch processor
        batch_processor = OptimizedBatchProcessor(batch_size=config['batch_size'])
        
        logger.info("Models initialized successfully")
        logger.info(f"Using detection threshold: {detector.threshold:.4f}")
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
            'batch_processor': batch_processor is not None,
            'dwt_model_loaded': dwt_model is not None,
            'dywt_model_loaded': dywt_model is not None
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

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    """
    Get information about available models
    Returns metadata about loaded models
    """
    try:
        # Basic model info
        models_info = {
            'detection': {
                'path': config['detection_model_path'],
                'filename': os.path.basename(config['detection_model_path']),
                'loaded': detector is not None,
                'threshold': config['threshold'],
                'type': 'RDLNN'
            },
            'localization': {
                'path': config['localization_model_path'],
                'filename': os.path.basename(config['localization_model_path']),
                'loaded': detector is not None,
                'type': 'PDyWTCNN Localizer'
            },
            'dwt': {
                'path': config['dwt_model_path'],
                'filename': os.path.basename(config['dwt_model_path']),
                'loaded': dwt_model is not None,
                'type': 'DWT'
            },
            'dywt': {
                'path': config['dywt_model_path'],
                'filename': os.path.basename(config['dywt_model_path']),
                'loaded': dywt_model is not None,
                'type': 'DyWT'
            }
        }
        
        # Add additional info if models are loaded
        if detector:
            # Get model specific details if available
            if hasattr(detector, 'rdlnn_model') and detector.rdlnn_model:
                rdlnn = detector.rdlnn_model
                models_info['detection']['input_dimension'] = rdlnn.model[0].in_features if hasattr(rdlnn.model[0], 'in_features') else 'unknown'
        
        return jsonify({
            'models': models_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/test_data/info', methods=['GET'])
def get_test_data_info():
    """
    Get information about available test data
    Returns metadata about test datasets
    """
    # This is a placeholder implementation
    return jsonify({
        'test_data': {
            'available': False,
            'message': 'No test data currently loaded'
        }
    })

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

@app.route('/api/compare', methods=['POST'])
def compare_images():
    """
    Compare two images for similarity and potential forgery
    Useful for detecting if one image was derived from or manipulated from another
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Two images are required for comparison'}), 400
    
    try:
        # Get the uploaded files
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        # Save the files temporarily
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(config['upload_folder'], filename1)
        filepath2 = os.path.join(config['upload_folder'], filename2)
        file1.save(filepath1)
        file2.save(filepath2)
        
        # Get base64 encoded images
        image1_base64 = image_to_base64(filepath1)
        image2_base64 = image_to_base64(filepath2)
        
        # Process each image individually
        start_time = time.time()
        result1 = detector.detect(filepath1)
        result2 = detector.detect(filepath2)
        
        # Calculate similarities between feature vectors
        if 'features' in result1 and 'features' in result2:
            features1 = result1['features']
            features2 = result2['features']
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = float(cosine_similarity([features1], [features2])[0][0])
        else:
            similarity = None
        
        # Create comparison results
        comparison_result = {
            'image1': {
                'filename': filename1,
                'result': 'forged' if result1['prediction'] == 1 else 'authentic',
                'probability': float(result1['probability'])
            },
            'image2': {
                'filename': filename2,
                'result': 'forged' if result2['prediction'] == 1 else 'authentic',
                'probability': float(result2['probability'])
            },
            'similarity': similarity,
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'image1_base64': image1_base64,
            'image2_base64': image2_base64
        }
        
        # Clean up temporary files
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except Exception as e:
            logger.warning(f"Failed to remove temporary files: {e}")
        
        return jsonify(comparison_result)
        
    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze/threshold', methods=['POST'])
def analyze_threshold():
    """
    Analyze the effect of different thresholds on prediction results
    Useful for finding the optimal threshold for a specific image
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
        
        # Get base64 encoded image
        image_base64 = image_to_base64(filepath)
        
        # Process the image
        start_time = time.time()
        result = detector.detect(filepath)
        
        # Get the raw probability
        probability = result.get('probability', 0.5)
        
        # Calculate results for different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_results = []
        
        for threshold in thresholds:
            threshold_results.append({
                'threshold': threshold,
                'prediction': 1 if probability >= threshold else 0,
                'result': 'forged' if probability >= threshold else 'authentic'
            })
        
        # Create response
        analysis_result = {
            'filename': filename,
            'original_probability': float(probability),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'image_base64': image_base64,
            'threshold_results': threshold_results,
            'current_threshold': detector.threshold,
            'current_result': 'forged' if probability >= detector.threshold else 'authentic'
        }
        
        # Clean up temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error in threshold analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/save_test_image', methods=['POST'])
def save_test_image():
    """
    Save an uploaded image as test data
    Useful for development and testing
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    try:
        # Get the uploaded file
        file = request.files['image']
        filename = secure_filename(file.filename)
        
        # Create test data directory if it doesn't exist
        test_data_dir = os.path.join(os.path.dirname(config['upload_folder']), 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Save the file with timestamp to ensure uniqueness
        timestamp = int(time.time())
        save_filename = f"{timestamp}_{filename}"
        save_path = os.path.join(test_data_dir, save_filename)
        file.save(save_path)
        
        # Get image information
        with Image.open(save_path) as img:
            image_info = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size
            }
        
        # Create response
        response = {
            'status': 'success',
            'message': 'Image saved successfully',
            'filename': save_filename,
            'path': save_path,
            'timestamp': timestamp,
            'image_info': image_info
        }
        
        logger.info(f"Test image saved: {save_path}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error saving test image: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/results/batch/<batch_id>', methods=['GET'])
def get_batch_results(batch_id):
    """
    Get results from a specific batch processing run
    Useful for retrieving results from previous batch operations
    """
    try:
        # This is a placeholder implementation
        # In a production system, you would retrieve this from a database
        
        return jsonify({
            'status': 'error',
            'message': 'Batch results retrieval is not implemented yet',
            'batch_id': batch_id
        }), 501
        
    except Exception as e:
        logger.error(f"Error retrieving batch results: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/run/detection', methods=['POST'])
def run_detection_script():
    """
    Run the main.py detection script directly with parameters
    Provides direct access to the underlying detection script
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
        
        # Get optional parameters
        threshold = request.form.get('threshold')
        if threshold:
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = None
        
        model_path = request.form.get('model_path')
        if not model_path or not os.path.exists(model_path):
            model_path = config['detection_model_path']
        
        # Run detection using detector with potentially custom parameters
        start_time = time.time()
        
        # Override threshold temporarily if provided
        original_threshold = detector.threshold
        if threshold is not None:
            detector.threshold = threshold
        
        # Perform detection
        result = detector.detect(filepath)
        
        # Restore original threshold
        if threshold is not None:
            detector.threshold = original_threshold
        
        # Format response with detailed information
        response = {
            'filename': filename,
            'result': 'forged' if result['prediction'] == 1 else 'authentic',
            'probability': float(result.get('probability', 0.5)),
            'threshold': float(threshold) if threshold is not None else float(original_threshold),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'model_path': model_path,
            'command_line_equivalent': f"python main.py --mode single --image_path {filename} --threshold {threshold if threshold is not None else original_threshold} --model_path {model_path}"
        }
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running detection script: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/run/localization', methods=['POST'])
def run_localization_script():
    """
    Run the main.py localization script directly with parameters
    Provides direct access to the underlying localization script
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
        
        # Get optional parameters
        threshold = request.form.get('threshold')
        if threshold:
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = None
        
        model_path = request.form.get('model_path')
        if not model_path or not os.path.exists(model_path):
            model_path = config['detection_model_path']
            
        localization_model_path = request.form.get('localization_model_path')
        if not localization_model_path or not os.path.exists(localization_model_path):
            localization_model_path = config['localization_model_path']
        
        # Override threshold temporarily if provided
        original_threshold = detector.threshold
        if threshold is not None:
            detector.threshold = threshold
        
        # First detect if image is forged
        start_time = time.time()
        detection_result = detector.detect(filepath)
        
        # Determine result type
        is_forged = detection_result['prediction'] == 1
        result_type = 'forged' if is_forged else 'authentic'
        
        # Base response with detection results
        response = {
            'filename': filename,
            'result': result_type,
            'probability': float(detection_result.get('probability', 0.5)),
            'threshold': float(threshold) if threshold is not None else float(original_threshold),
            'processing_time': 0,
            'timestamp': int(time.time()),
            'model_path': model_path,
            'localization_model_path': localization_model_path
        }
        
        # If forged, perform localization
        if is_forged:
            # Create a temporary file path for the forgery map
            run_id = f"run_{int(time.time())}"
            forgery_map_path = os.path.join(config['upload_folder'], f"temp_forgery_map_{run_id}.png")
            
            # Localize the forgery
            localization_result = detector.localize(filepath, save_path=forgery_map_path)
            
            # Convert forgery map to base64
            forgery_map_base64 = image_to_base64(forgery_map_path, format='PNG')
            
            # Update the response with localization information
            response['forgery_map_base64'] = forgery_map_base64
            response['regions'] = localization_result.get('region_proposals', [])
            response['region_count'] = len(localization_result.get('region_proposals', []))
            
            # Remove temporary forgery map file
            try:
                os.remove(forgery_map_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary forgery map {forgery_map_path}: {e}")
        
        # Update processing time
        processing_time = time.time() - start_time
        response['processing_time'] = processing_time
        
        # Add command line equivalent
        response['command_line_equivalent'] = (
            f"python main.py --mode localize --image_path {filename} "
            f"--threshold {threshold if threshold is not None else original_threshold} "
            f"--model_path {model_path} "
            f"--localization_model_path {localization_model_path}"
        )
        
        # Restore original threshold
        if threshold is not None:
            detector.threshold = original_threshold
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error running localization script: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/api/dwt/detect', methods=['POST'])
def dwt_detect():
    """Detect forgery using the DWT model"""
    if not dwt_model:
        return jsonify({'error': 'DWT model not initialized'}), 500
    
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
        prediction = dwt_detect_forgery(dwt_model, filepath)
        end_time = time.time()
        
        # Get the original image as base64
        original_image_base64 = image_to_base64(filepath)
        
        # Determine result
        if prediction is None:
            return jsonify({
                'error': 'Could not process image',
                'filename': filename
            }), 500
            
        result_type = 'forged' if prediction == 1 else 'authentic'
        
        # Format the result data
        result_data = {
            'image': filename,
            'result': result_type,
            'prediction': int(prediction),
            'processing_time': f"{end_time - start_time:.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': 'DWT'
        }
        
        # Format result text
        result_text = format_dict_to_text(result_data)
        
        # Create response
        run_id = f"run_{int(time.time())}"
        response = {
            'run_id': run_id,
            'filename': filename,
            'result': result_type,
            'prediction': int(prediction),
            'processing_time': end_time - start_time,
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
        logger.error(f"Error in DWT forgery detection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/dywt/detect', methods=['POST'])
def dywt_detect():
    """Detect forgery using the DyWT model"""
    if not dywt_model:
        return jsonify({'error': 'DyWT model not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Get threshold parameter if provided
        threshold = request.form.get('threshold')
        if threshold:
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = 0.5
        else:
            threshold = 0.5
        
        # Process the image
        start_time = time.time()
        prediction, probability = dywt_detect_forgery(dywt_model, filepath, threshold=threshold)
        end_time = time.time()
        
        # Get the original image as base64
        original_image_base64 = image_to_base64(filepath)
        
        # Determine result
        if prediction is None:
            return jsonify({
                'error': 'Could not process image',
                'filename': filename
            }), 500
            
        result_type = 'forged' if prediction == 1 else 'authentic'
        
        # Format the result data
        result_data = {
            'image': filename,
            'result': result_type,
            'prediction': int(prediction),
            'probability': f"{float(probability):.4f}",
            'threshold': f"{float(threshold):.4f}",
            'processing_time': f"{end_time - start_time:.4f}",
            'timestamp': datetime.now().isoformat(),
            'model': 'DyWT'
        }
        
        # Format result text
        result_text = format_dict_to_text(result_data)
        
        # Create response
        run_id = f"run_{int(time.time())}"
        response = {
            'run_id': run_id,
            'filename': filename,
            'result': result_type,
            'prediction': int(prediction),
            'probability': float(probability),
            'threshold': float(threshold),
            'processing_time': end_time - start_time,
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
        logger.error(f"Error in DyWT forgery detection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/rdlnn/detect', methods=['POST'])
def rdlnn_detect():
    """
    Detect forgery using the RDLNN model specifically
    Similar to the regular detect endpoint but ensures RDLNN model is used
    """
    if not detector:
        return jsonify({'error': 'Models not initialized'}), 500
    
    # Verify RDLNN model is available
    if not hasattr(detector, 'rdlnn_model') or detector.rdlnn_model is None:
        logger.error("RDLNN model not available for detection")
        return jsonify({'error': 'RDLNN model not properly initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['upload_folder'], filename)
        file.save(filepath)
        
        # Log file information
        logger.info(f"Processing file: {filename}, size: {os.path.getsize(filepath)}")
        
        # Get optional threshold
        threshold = request.form.get('threshold')
        if threshold:
            try:
                threshold = float(threshold)
                # Override threshold temporarily
                original_threshold = detector.threshold
                detector.threshold = threshold
                logger.info(f"Using custom threshold from request: {threshold}")
            except ValueError:
                threshold = None
        else:
            original_threshold = detector.threshold
            threshold = original_threshold
            logger.info(f"Using default threshold: {threshold}")
        
        # Process the image
        start_time = time.time()
        
        # Extract features using the comprehensive method
        feature_vector = detector.extract_features(filepath)
        
        if feature_vector is None:
            logger.error(f"Failed to extract features from {filepath}")
            return jsonify({
                'error': 'Failed to extract features from image'
            }), 500
            
        # Log feature vector shape
        logger.info(f"Extracted feature vector with shape: {feature_vector.shape}")
        
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Apply feature selection if available
        if hasattr(detector.rdlnn_model, 'feature_selector') and detector.rdlnn_model.feature_selector is not None:
            try:
                logger.info(f"Applying feature selection, current shape: {feature_vector.shape}")
                feature_vector = detector.rdlnn_model.feature_selector(feature_vector)
                logger.info(f"After feature selection, shape: {feature_vector.shape}")
            except Exception as e:
                logger.error(f"Error applying feature selection: {e}")
                logger.error(traceback.format_exc())
        
        # Get input dimension expected by the model
        input_dim = detector.rdlnn_model.model[0].in_features
        logger.info(f"Model expects input dimension: {input_dim}")
        
        # Ensure feature vector has the right dimension
        if feature_vector.shape[1] != input_dim:
            logger.warning(f"Feature dimension mismatch: got {feature_vector.shape[1]}, expected {input_dim}")
            
            if feature_vector.shape[1] > input_dim:
                logger.info(f"Truncating features from {feature_vector.shape[1]} to {input_dim}")
                feature_vector = feature_vector[:, :input_dim]
            else:
                logger.info(f"Padding features from {feature_vector.shape[1]} to {input_dim}")
                padded = np.zeros((1, input_dim))
                padded[:, :feature_vector.shape[1]] = feature_vector
                feature_vector = padded
        
        # Make prediction with RDLNN model
        logger.info("Making prediction with RDLNN model")
        predictions, probabilities = detector.rdlnn_model.predict(feature_vector)
        
        # Get the prediction results
        prediction = int(predictions[0])
        probability = float(probabilities[0])
        
        # Determine result type based on threshold
        is_forged = probability >= threshold
        result_type = 'forged' if is_forged else 'authentic'
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Result: {result_type}")
        
        # Format the response
        response = {
            'filename': filename,
            'result': result_type,
            'probability': probability,
            'threshold': float(threshold),
            'processing_time': time.time() - start_time,
            'timestamp': int(time.time()),
            'model': 'RDLNN'
        }
        
        # Restore original threshold if it was changed
        if threshold != original_threshold:
            detector.threshold = original_threshold
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in RDLNN detection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/rdlnn/localize', methods=['POST'])
def rdlnn_localize():
    """
    Detect and localize forgery using the RDLNN model with PDyWT localizer
    Similar to the regular localize endpoint but ensures RDLNN model is used for detection
    """
    # Implementation is similar to the regular localize endpoint but with RDLNN-specific logic
    # For brevity, the implementation is omitted as it would be very similar to the rdlnn_detect function
    # combined with the localize_forgery function
    
    return jsonify({
        'status': 'error',
        'message': 'RDLNN localization endpoint is not implemented yet'
    }), 501

@app.route('/api/rdlnn/debug', methods=['POST'])
def rdlnn_debug():
    """
    Debug endpoint that provides detailed information about the RDLNN processing pipeline
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
        
        # Debug info container
        debug_info = {
            'image': {
                'filename': filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
                'mime_type': file.content_type
            },
            'detector': {
                'has_rdlnn': hasattr(detector, 'rdlnn_model') and detector.rdlnn_model is not None,
                'threshold': detector.threshold
            },
            'processing_steps': []
        }
        
        # Check if RDLNN model is available
        if hasattr(detector, 'rdlnn_model') and detector.rdlnn_model is not None:
            model = detector.rdlnn_model
            debug_info['rdlnn_model'] = {
                'input_dim': model.model[0].in_features,
                'has_feature_selector': hasattr(model, 'feature_selector') and model.feature_selector is not None,
                'has_threshold': hasattr(model, 'threshold'),
                'threshold': getattr(model, 'threshold', 0.5)
            }
        
        # Step 1: Extract features
        debug_info['processing_steps'].append({
            'step': 'Feature extraction',
            'start_time': time.time()
        })
        
        feature_vector = detector.extract_features(filepath)
        
        if feature_vector is None:
            debug_info['processing_steps'][-1]['error'] = 'Failed to extract features'
            return jsonify(debug_info), 500
        
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        debug_info['processing_steps'][-1].update({
            'end_time': time.time(),
            'feature_shape': feature_vector.shape,
            'feature_stats': {
                'min': float(np.min(feature_vector)),
                'max': float(np.max(feature_vector)),
                'mean': float(np.mean(feature_vector)),
                'std': float(np.std(feature_vector))
            }
        })
        
        # Step 2: Apply feature selection if available
        debug_info['processing_steps'].append({
            'step': 'Feature selection',
            'start_time': time.time()
        })
        
        if hasattr(detector.rdlnn_model, 'feature_selector') and detector.rdlnn_model.feature_selector is not None:
            try:
                original_shape = feature_vector.shape
                feature_vector = detector.rdlnn_model.feature_selector(feature_vector)
                
                debug_info['processing_steps'][-1].update({
                    'end_time': time.time(),
                    'original_shape': original_shape,
                    'new_shape': feature_vector.shape,
                    'feature_stats_after': {
                        'min': float(np.min(feature_vector)),
                        'max': float(np.max(feature_vector)),
                        'mean': float(np.mean(feature_vector)),
                        'std': float(np.std(feature_vector))
                    }
                })
            except Exception as e:
                debug_info['processing_steps'][-1].update({
                    'end_time': time.time(),
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        else:
            debug_info['processing_steps'][-1].update({
                'end_time': time.time(),
                'skipped': True,
                'reason': 'No feature selector available'
            })
        
        # Step 3: Make prediction
        debug_info['processing_steps'].append({
            'step': 'Prediction',
            'start_time': time.time()
        })
        
        try:
            # Make prediction
            predictions, probabilities = detector.rdlnn_model.predict(feature_vector)
            
            debug_info['processing_steps'][-1].update({
                'end_time': time.time(),
                'raw_predictions': predictions.tolist(),
                'raw_probabilities': probabilities.tolist(),
                'threshold': float(detector.threshold)
            })
            
            # Add processed results
            debug_info['results'] = {
                'prediction': int(predictions[0]),
                'probability': float(probabilities[0]),
                'result_type': 'forged' if predictions[0] == 1 else 'authentic',
                'threshold': float(detector.threshold)
            }
            
        except Exception as e:
            debug_info['processing_steps'][-1].update({
                'end_time': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        
        # Clean up the temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            debug_info['cleanup_error'] = str(e)
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({
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
    parser.add_argument('--detection-model', type=str, default='./data/models/rdlnn_model.pth',
                        help='Path to detection model')
    parser.add_argument('--localization-model', type=str, default='./data/models/localizer.pth',
                        help='Path to localization model')
    
    # Directories
    parser.add_argument('--upload-folder', type=str, default='./uploads',
                        help='Temporary folder for uploaded images')
    
    # Processing parameters
    parser.add_argument('--threshold', type=float, default=0.55,
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