#!/usr/bin/env python3
"""
Forgery localization interface to integrate with the existing forgery detection system.
This module provides functions to use the PDyWT-CNN detector for both detection and localization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the main detector
from modules.pdywt_cnn import PDyWTCNNDetector
from modules.utils import logger


def analyze_image(image_path, model_path=None, localization_model_path=None, output_dir='data/results'):
    """
    Analyze an image for forgery detection and localization
    
    Args:
        image_path: Path to the image file
        model_path: Path to pretrained detection model
        localization_model_path: Path to pretrained localization model
        output_dir: Directory to save results
        
    Returns:
        Dictionary with detection and localization results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the detector
    detector = PDyWTCNNDetector(
        model_path=model_path,
        localization_model_path=localization_model_path
    )
    
    # Perform forgery detection
    detection_result = detector.detect(image_path)
    
    # Initialize result dictionary
    result = {
        'image_path': image_path,
        'prediction': detection_result.get('prediction', None),
        'probability': detection_result.get('probability', None),
        'error': detection_result.get('error', None)
    }
    
    # If detection indicates forgery, perform localization
    if detection_result.get('prediction', 0) == 1:
        # Generate output filepath
        base_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_forgery_map.png")
        
        # Run localization
        localization_result = detector.localize(image_path, save_path=output_path)
        
        # Add localization results to the output
        result.update({
            'localization_heatmap': localization_result.get('heatmap', None),
            'region_proposals': localization_result.get('region_proposals', []),
            'localization_map_path': output_path if not 'error' in localization_result else None,
            'localization_error': localization_result.get('error', None)
        })
    
    return result


def batch_process_directory(input_dir, model_path=None, localization_model_path=None, output_dir='data/results'):
    """
    Process all images in a directory for forgery detection and localization
    
    Args:
        input_dir: Directory containing image files
        model_path: Path to pretrained detection model
        localization_model_path: Path to pretrained localization model
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results for each image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # Process each image
    results = {}
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        logger.info(f"Processing {image_path}")
        
        try:
            # Analyze image
            result = analyze_image(
                image_path, 
                model_path=model_path,
                localization_model_path=localization_model_path,
                output_dir=output_dir
            )
            
            # Store result
            results[image_file] = result
            
            # Log result
            if result.get('error'):
                logger.error(f"Error analyzing {image_file}: {result['error']}")
            else:
                prediction = "FORGED" if result['prediction'] == 1 else "AUTHENTIC"
                confidence = result['probability']
                logger.info(f"Result for {image_file}: {prediction} (Confidence: {confidence:.4f})")
                
                if prediction == "FORGED" and 'region_proposals' in result:
                    logger.info(f"Found {len(result['region_proposals'])} suspicious regions.")
        
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            results[image_file] = {'error': str(e)}
    
    # Create a summary report
    create_summary_report(results, output_dir)
    
    return results


def create_summary_report(results, output_dir):
    """
    Create a summary report of batch processing results
    
    Args:
        results: Dictionary with results for each image
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, 'forgery_detection_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Image Forgery Detection and Localization Report\n")
        f.write("=============================================\n\n")
        
        # Count statistics
        total_images = len(results)
        forged_images = sum(1 for result in results.values() 
                           if not 'error' in result and result.get('prediction') == 1)
        authentic_images = sum(1 for result in results.values() 
                             if not 'error' in result and result.get('prediction') == 0)
        error_images = sum(1 for result in results.values() if 'error' in result)
        
        # Write summary
        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Detected as forged: {forged_images}\n")
        f.write(f"Detected as authentic: {authentic_images}\n")
        f.write(f"Errors: {error_images}\n\n")
        
        # Write detailed results
        f.write("Detailed Results:\n")
        f.write("-----------------\n\n")
        
        for image_name, result in sorted(results.items()):
            f.write(f"Image: {image_name}\n")
            
            if 'error' in result and result['error']:
                f.write(f"  Error: {result['error']}\n")
            else:
                prediction = "FORGED" if result['prediction'] == 1 else "AUTHENTIC"
                f.write(f"  Result: {prediction}\n")
                f.write(f"  Confidence: {result['probability']:.4f}\n")
                
                if prediction == "FORGED" and 'region_proposals' in result:
                    f.write(f"  Suspicious regions: {len(result['region_proposals'])}\n")
                    if result.get('localization_map_path'):
                        f.write(f"  Localization map: {os.path.basename(result['localization_map_path'])}\n")
            
            f.write("\n")
    
    logger.info(f"Summary report saved to {report_path}")


def train_forgery_models(features_path, annotations_dir, model_output_path, 
                      localization_model_output_path, data_dir=None):
    """
    Train both forgery detection and localization models
    
    Args:
        features_path: Path to feature file for detection model training
        annotations_dir: Directory containing annotation masks for localization training
        model_output_path: Path to save detection model
        localization_model_output_path: Path to save localization model
        data_dir: Directory containing image files for localization training (if None, extracted from features_path)
    """
    # Initialize the detector
    detector = PDyWTCNNDetector()
    
    # Train detection model if features_path is provided
    if features_path:
        logger.info(f"Training detection model using features from {features_path}")
        detector.train_detector(
            features_path=features_path,
            labels=None,  # Assume labels are included in the features file
            output_path=model_output_path,
            epochs=25,
            batch_size=32,
            learning_rate=0.001
        )
    
    # Train localization model if annotations_dir is provided
    if annotations_dir and data_dir:
        logger.info(f"Training localization model using data from {data_dir} and annotations from {annotations_dir}")
        detector.train_localizer(
            data_dir=data_dir,
            annotations_dir=annotations_dir,
            output_path=localization_model_output_path,
            epochs=30,
            batch_size=8,
            learning_rate=0.0005
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Forgery Detection and Localization")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze single image command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single image')
    analyze_parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    analyze_parser.add_argument('--model', type=str, default=None, help='Path to detection model')
    analyze_parser.add_argument('--localization_model', type=str, default=None, help='Path to localization model')
    analyze_parser.add_argument('--output_dir', type=str, default='data/results', help='Directory to save results')
    
    # Batch process directory command
    batch_parser = subparsers.add_parser('batch', help='Process all images in a directory')
    batch_parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
    batch_parser.add_argument('--model', type=str, default=None, help='Path to detection model')
    batch_parser.add_argument('--localization_model', type=str, default=None, help='Path to localization model')
    batch_parser.add_argument('--output_dir', type=str, default='data/results', help='Directory to save results')
    
    # Train models command
    train_parser = subparsers.add_parser('train', help='Train forgery detection and localization models')
    train_parser.add_argument('--features', type=str, required=True, help='Path to features file for detection training')
    train_parser.add_argument('--annotations_dir', type=str, required=True, help='Directory with annotation masks')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Directory with original images')
    train_parser.add_argument('--model_output', type=str, default='data/models/pdywt_detector.pth', help='Output path for detection model')
    train_parser.add_argument('--localization_model_output', type=str, default='data/models/pdywt_localizer.pth', help='Output path for localization model')
    
    args = parser.parse_args()
    
    # Execute the chosen command
    if args.command == 'analyze':
        result = analyze_image(
            args.image,
            model_path=args.model,
            localization_model_path=args.localization_model,
            output_dir=args.output_dir
        )
        
        # Print results
        if 'error' in result and result['error']:
            print(f"Error: {result['error']}")
        else:
            prediction = "FORGED" if result['prediction'] == 1 else "AUTHENTIC"
            print(f"Result: {prediction}")
            print(f"Confidence: {result['probability']:.4f}")
            
            if prediction == "FORGED" and 'region_proposals' in result:
                print(f"Found {len(result['region_proposals'])} suspicious regions.")
                if result.get('localization_map_path'):
                    print(f"Localization map saved to: {result['localization_map_path']}")
    
    elif args.command == 'batch':
        batch_process_directory(
            args.input_dir,
            model_path=args.model,
            localization_model_path=args.localization_model,
            output_dir=args.output_dir
        )
    
    elif args.command == 'train':
        train_forgery_models(
            features_path=args.features,
            annotations_dir=args.annotations_dir,
            model_output_path=args.model_output,
            localization_model_output_path=args.localization_model_output,
            data_dir=args.data_dir
        )
    
    else:
        parser.print_help()