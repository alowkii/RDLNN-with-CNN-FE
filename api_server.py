#!/usr/bin/env python3
"""
RESTful API server for image forgery detection system
Provides endpoints for detection and localization of forged images
"""

import os
import time
import argparse
import numpy as np
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import logging

# Import the image forgery detection modules
from modules.rdlnn import RegressionDLNN
from modules.feature_extractor import PDyWTCNNDetector
from modules.utils import setup_logging, logger

# Initialize app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Global variables
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB maximum file size

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize detector (will be set in main)
detector = None

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'time': time.time(),
        'service': 'Image Forgery Detection API',
        'model_info': {
            'name': 'PDyWT-CNN',
            'threshold': getattr(detector, 'threshold', 0.7) if detector else 0.7
        }
    })

@app.route('/api/detect', methods=['POST'])
def detect_image():
    """
    Detect if an image is forged
    
    Expects a file upload with key 'image'
    Returns detection result with confidence
    """
    # Check if detector is initialized
    if detector is None:
        return jsonify({
            'error': 'Detection service not initialized'
        }), 500
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided'
        }), 400
    
    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({
            'error': 'Empty file provided'
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
    file.save(file_path)
    
    app.logger.info(f"Processing image: {file_path}")
    
    try:
        # Perform forgery detection
        detection_result = detector.detect(file_path)
        
        if 'error' in detection_result:
            return jsonify({
                'error': detection_result['error']
            }), 500
        
        # Format the response
        result = {
            'result': 'forged' if detection_result['prediction'] == 1 else 'authentic',
            'probability': float(detection_result['probability']),
            'timestamp': timestamp,
            'filename': filename
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error during detection: {str(e)}")
        return jsonify({
            'error': f"Detection failed: {str(e)}"
        }), 500

@app.route('/api/localize', methods=['POST'])
def localize_forgery():
    """
    Localize forgery regions in an image
    
    Expects a file upload with key 'image'
    Returns forgery map and locations of suspicious regions
    """
    # Check if detector is initialized
    if detector is None:
        return jsonify({
            'error': 'Localization service not initialized'
        }), 500
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided'
        }), 400
    
    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({
            'error': 'Empty file provided'
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
    file.save(file_path)
    
    app.logger.info(f"Localizing forgery in image: {file_path}")
    
    try:
        # First check if the image is forged
        detection_result = detector.detect(file_path)
        
        if 'error' in detection_result:
            return jsonify({
                'error': detection_result['error']
            }), 500
        
        # Get the probability and apply threshold
        probability = detection_result['probability']
        threshold = getattr(detector, 'threshold', 0.7)
        is_forged = detection_result['prediction'] == 1
        
        # Only perform localization if the image is classified as forged
        if is_forged:
            # Generate result filename
            result_filename = f"{timestamp}_forgery_map.png"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            # Perform forgery localization
            localization_result = detector.localize(file_path, save_path=result_path)
            
            if 'error' in localization_result:
                return jsonify({
                    'error': localization_result['error']
                }), 500
            
            # Format the response
            result = {
                'result': 'forged',
                'probability': float(probability),
                'timestamp': timestamp,
                'filename': filename,
                'forgery_map': f"/api/results/{result_filename}",
                'regions': localization_result['region_proposals'],
                'threshold': float(threshold)
            }
            
            return jsonify(result)
        else:
            # Image is authentic, no need for localization
            return jsonify({
                'result': 'authentic',
                'probability': float(probability),
                'timestamp': timestamp,
                'filename': filename,
                'message': 'Image appears authentic, no forgery localization performed.',
                'threshold': float(threshold)
            })
        
    except Exception as e:
        app.logger.error(f"Error during localization: {str(e)}")
        return jsonify({
            'error': f"Localization failed: {str(e)}"
        }), 500

@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    """
    Retrieve a result file (e.g., forgery map image)
    """
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/api/batch/detect', methods=['POST'])
def batch_detect():
    """
    Process multiple images for forgery detection
    
    Expects multiple file uploads with key 'images[]'
    Returns detection results for all images
    """
    # Check if detector is initialized
    if detector is None:
        return jsonify({
            'error': 'Detection service not initialized'
        }), 500
    
    # Check if images were uploaded
    if 'images[]' not in request.files:
        return jsonify({
            'error': 'No images provided'
        }), 400
    
    files = request.files.getlist('images[]')
    
    # Check if files array is empty
    if len(files) == 0:
        return jsonify({
            'error': 'No files provided'
        }), 400
    
    threshold = getattr(detector, 'threshold', 0.7)
    results = []
    for file in files:
        # Check file extension
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            })
            continue
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(file_path)
        
        app.logger.info(f"Processing image in batch: {file_path}")
        
        try:
            # Perform forgery detection
            detection_result = detector.detect(file_path)
            
            if 'error' in detection_result:
                results.append({
                    'filename': filename,
                    'error': detection_result['error']
                })
            else:
                # Format the response
                results.append({
                    'filename': filename,
                    'result': 'forged' if detection_result['prediction'] == 1 else 'authentic',
                    'probability': float(detection_result['probability']),
                    'timestamp': timestamp,
                    'threshold': float(threshold)
                })
                
        except Exception as e:
            app.logger.error(f"Error during detection of {filename}: {str(e)}")
            results.append({
                'filename': filename,
                'error': f"Detection failed: {str(e)}"
            })
    
    return jsonify({
        'batch_results': results,
        'total': len(files),
        'successful': sum(1 for r in results if 'error' not in r),
        'threshold': float(threshold)
    })

def cleanup_old_files():
    """Periodic cleanup of old uploaded files and results"""
    while True:
        try:
            current_time = time.time()
            # Delete files older than 1 hour
            expiration_time = current_time - (60 * 60)
            
            # Clean uploads folder
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < expiration_time:
                    os.remove(file_path)
                    app.logger.info(f"Removed old upload: {filename}")
            
            # Clean results folder
            for filename in os.listdir(RESULTS_FOLDER):
                file_path = os.path.join(RESULTS_FOLDER, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < expiration_time:
                    os.remove(file_path)
                    app.logger.info(f"Removed old result: {filename}")
                    
        except Exception as e:
            app.logger.error(f"Error during cleanup: {str(e)}")
        
        # Sleep for 15 minutes before next cleanup
        time.sleep(15 * 60)

def main():
    """Main function to start the server"""
    global detector
    
    parser = argparse.ArgumentParser(description='Image Forgery Detection API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model_path', type=str, default='data/models/forgery_detection_model.pth',
                        help='Path to the forgery detection model')
    parser.add_argument('--localization_model_path', type=str, default='data/models/pdywt_localizer.pth',
                        help='Path to the forgery localization model')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom threshold for forgery detection (default: use model threshold or 0.7)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(RESULTS_FOLDER)
    
    # Initialize the detector
    app.logger.info(f"Initializing detector with model: {args.model_path}")
    try:
        detector = PDyWTCNNDetector(
            model_path=args.model_path,
            localization_model_path=args.localization_model_path,
            use_gpu=True
        )
        
        # Override threshold if provided
        if args.threshold is not None:
            detector.threshold = args.threshold
            app.logger.info(f"Using custom threshold: {args.threshold}")
        else:
            app.logger.info(f"Using model threshold: {detector.threshold}")
            
        app.logger.info("Detector initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize detector: {str(e)}")
        return
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    
    # Print API information
    threshold_info = f"Using detection threshold: {detector.threshold}"
    
    print("\n" + "="*50)
    print("Image Forgery Detection API Server")
    print("="*50)
    print(f"Server running at: http://{args.host}:{args.port}")
    print(threshold_info)
    print("Available endpoints:")
    print("  GET  /api/health                - Health check")
    print("  POST /api/detect                - Detect if an image is forged")
    print("  POST /api/localize              - Localize forgery in an image")
    print("  GET  /api/results/<filename>    - Get result file")
    print("  POST /api/batch/detect          - Batch detection of multiple images")
    print("="*50 + "\n")
    
    # Start the Flask server
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()