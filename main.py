#!/usr/bin/env python3
"""
Image Forgery Detection System - Main entry point
Provides a CLI interface for different modes of operation
"""

import os
import argparse
import torch
import time
import signal
import sys
import gc
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch.nn.functional as F
import tqdm

# Import modules
# Import modules
from modules.rdlnn import RegressionDLNN
from modules.data_handling import precompute_features, load_and_verify_features
from modules.preprocessing import preprocess_image
from modules.feature_extractor import PDyWTCNNDetector
from modules.batch_processor import OptimizedBatchProcessor
from modules.utils import setup_signal_handlers, plot_training_history, setup_logging, logger

# Import training methods
from training.balanced import train_with_balanced_sampling, train_with_oversampling, combined_approach
from training.precision import precision_tuned_training

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.features_path), exist_ok=True)
    
    # Set up logging
    setup_logging(args.output_dir)
    logger.info(f"Starting image forgery detection in {args.mode} mode")
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} - CUDA available: {torch.cuda.is_available()}")
    
    # Execute the requested mode
    if args.mode == 'precompute':
        precompute_mode(args)
    elif args.mode == 'localize':
        localize_mode(args)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'test':
        test_mode(args)
    elif args.mode == 'single':
        test_single_image(args)
    elif args.mode == 'analyze':
        analyze_features(args)
    
    logger.info("Operation completed successfully")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Image Forgery Detection System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--localization_model_path', 
                        type=str, 
                        default='data/models/pdywt_localizer.pth',
                        help='Path to localization model')
                    
    parser.add_argument('--mode', 
                        choices=['train', 'test', 'precompute', 'single', 'analyze', 'localize'], 
                        required=True, 
                        help='Operating mode: train, test, precompute, single, analyze, or localize')
    
    parser.add_argument('--input_dir', 
                        type=str,
                        help='Directory containing input images')
    
        
    parser.add_argument('--image_path', 
                        type=str,
                        help='Path to single image for testing')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='data/results',
                        help='Directory to save results')
    
    parser.add_argument('--model_path', 
                        type=str, 
                        default='data/models/forgery_detection_model.pth',
                        help='Path to save/load model')
    
    parser.add_argument('--features_path', 
                        type=str, 
                        default='data/features/precomputed_features.npz',
                        help='Path to save/load precomputed features')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=16,
                        help='Batch size for processing')
    
    parser.add_argument('--workers', 
                        type=int, 
                        default=4,
                        help='Number of worker threads for data loading')
    
    parser.add_argument('--fp16', 
                        action='store_true',
                        help='Use half precision (FP16) operations')
    
    parser.add_argument('--authentic_dir', 
                        type=str,
                        help='Directory containing authentic images (for training)')
    
    parser.add_argument('--forged_dir', 
                        type=str,
                        help='Directory containing forged images (for training)')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=30,
                        help='Number of training epochs')
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.003,
                        help='Learning rate for training')
    
    parser.add_argument('--training_method',
                        type=str,
                        choices=['balanced', 'oversampling', 'combined', 'precision'],
                        default='precision',
                        help='Training method to use')
    
    parser.add_argument('--threshold',
                        type=float,
                        default=None,
                        help='Override classification threshold (default: use model\'s threshold or 0.675)')
    
    parser.add_argument('--debug', 
                        action='store_true', 
                        help='Enable debug mode with additional logging')
    
    parser.add_argument('--localize', 
                        action='store_true',
                        help='Perform forgery localization on detected forgeries')

    parser.add_argument('--focal_gamma', 
                        type=float, 
                        default=2.0,
                        help='Gamma parameter for focal loss')

    parser.add_argument('--pos_weight', 
                        type=float, 
                        default=10.0,
                        help='Weight for positive class in loss function')

    parser.add_argument('--minority_ratio', 
                        type=float, 
                        default=0.8,
                        help='Ratio of minority class samples after resampling')

    parser.add_argument('--feature_selection', 
                        type=str,
                        choices=['none', 'variance', 'mutual_info', 'random_forest'],
                        default='random_forest',
                        help='Feature selection method')

    parser.add_argument('--feature_ratio', 
                        type=float,
                        default=0.8,
                        help='Ratio of features to select (0.0-1.0)')

    return parser.parse_args()

def precompute_mode(args: argparse.Namespace) -> None:
    """Precompute features for training or testing
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Precomputing features with batch size {args.batch_size}...")
    
    # Process authentic images if provided
    if args.authentic_dir:
        logger.info(f"Processing authentic images from {args.authentic_dir}")
        authentic_features = precompute_features(
            directory=args.authentic_dir,
            label=0,  # 0 = authentic
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=f"{os.path.splitext(args.features_path)[0]}_authentic.npz"
        )
    
    # Process forged images if provided
    if args.forged_dir:
        logger.info(f"Processing forged images from {args.forged_dir}")
        forged_features = precompute_features(
            directory=args.forged_dir,
            label=1,  # 1 = forged
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=f"{os.path.splitext(args.features_path)[0]}_forged.npz"
        )
    
    # Process general input directory if provided
    if args.input_dir and args.input_dir != args.authentic_dir and args.input_dir != args.forged_dir:
        logger.info(f"Processing images from {args.input_dir}")
        precompute_features(
            directory=args.input_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=args.features_path
        )
    
    # Combine authentic and forged features if both were computed
    if args.authentic_dir and args.forged_dir:
        logger.info("Combining authentic and forged features...")
        
        # Import combine_features function
        from tools.combine_features import combine_feature_files
        
        # Combine the files
        authentic_path = f"{os.path.splitext(args.features_path)[0]}_authentic.npz"
        forged_path = f"{os.path.splitext(args.features_path)[0]}_forged.npz"
        
        combine_feature_files(authentic_path, forged_path, args.features_path)

def train_mode(args: argparse.Namespace) -> None:
    """Train the model using precomputed features
    
    Args:
        args: Command line arguments
    """
    if not (args.authentic_dir and args.forged_dir) and not os.path.exists(args.features_path):
        logger.error("For training, either provide authentic_dir and forged_dir arguments, "
              "or precompute features first and provide features_path.")
        return
    
    # Load or compute features
    if not os.path.exists(args.features_path):
        logger.info("Precomputing features for training...")
        precompute_mode(args)
    
    # Load features
    features, labels, paths = load_and_verify_features(args.features_path)
    
    # Apply feature selection
    from modules.data_handling import perform_feature_selection
    logger.info("Performing feature selection...")
    selected_features, feature_selector = perform_feature_selection(
        features, labels, method='random_forest', n_features=int(features.shape[1] * 0.8)
    )
    
    logger.info(f"Selected {selected_features.shape[1]} features out of {features.shape[1]}")
    
    # Save selected features
    selected_features_path = os.path.splitext(args.features_path)[0] + "_selected.npz"
    np.savez(
        selected_features_path,
        features=selected_features,
        labels=labels,
        paths=paths
    )
    logger.info(f"Selected features saved to {selected_features_path}")
    
    # Use the selected training method with selected features
    if args.training_method == 'balanced':
        logger.info("Using balanced training method with weighted sampling")
        train_with_balanced_sampling(
            selected_features_path,
            args.model_path,
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size
        )
        
    elif args.training_method == 'oversampling':
        logger.info("Using oversampling training method")
        train_with_oversampling(
            args.features_path,
            args.model_path,
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size
        )
    elif args.training_method == 'combined':
        logger.info("Using combined training method (oversampling + class weights)")
        combined_approach(
            args.features_path,
            args.model_path,
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size
        )
    elif args.training_method == 'precision':
        logger.info(f"Using precision-tuned training method with threshold {args.threshold}")
        model, X_val, y_val = precision_tuned_training(
            args.features_path,
            args.model_path,
            args.output_dir,
            minority_ratio=args.minority_ratio,
            pos_weight=args.pos_weight,
            focal_gamma=args.focal_gamma,
            epochs=args.epochs, 
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            threshold=args.threshold
        )
        
        logger.info(f"Model saved with threshold to {args.model_path}")

def test_mode(args: argparse.Namespace) -> None:
    """Test the model on images in the input directory
    
    Args:
        args: Command line arguments
    """
    import numpy as np
    
    if not os.path.exists(args.model_path):
        logger.error(f"Error: Model not found at {args.model_path}")
        return
    
    if not args.input_dir:
        logger.error("Error: Please provide --input_dir argument for testing")
        return
    
    # Load the model
    model = RegressionDLNN.load(args.model_path)
    
    # Get the expected feature dimension from the model
    input_dim = model.model[0].in_features
    logger.info(f"Model expects {input_dim} features")
    
    # Check if model has a custom threshold
    threshold = getattr(model, 'threshold', 0.675)
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Using override threshold: {threshold}")
    else:
        logger.info(f"Using model's threshold: {threshold}")
    
    # Get image files
    image_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    num_images = len(image_files)
    logger.info(f"Found {num_images} images in {args.input_dir}")
    
    # Process results
    results = {
        'authentic': [],
        'forged': [],
        'errors': []
    }
    
    # Create detailed results output
    results_file = os.path.join(args.output_dir, 'detection_results.txt')
    with open(results_file, 'w') as f:
        f.write("Image Forgery Detection Results\n")
        f.write("==============================\n\n")
        f.write(f"Using threshold: {threshold:.4f}\n\n")
        
        # Process each image
        detector = PDyWTCNNDetector()
        
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(args.input_dir, image_file)
            
            try:
                # Extract features - simplify to use just wavelet features
                ycbcr_tensor = detector.preprocess_image(image_path)
                
                if ycbcr_tensor is None:
                    f.write(f"Image: {image_file}\n")
                    f.write(f"Result: ERROR - Could not preprocess image\n\n")
                    results['errors'].append(image_file)
                    continue
                    
                # Extract wavelet features
                feature_tensor = detector.extract_wavelet_features(ycbcr_tensor)
                
                # Get a fixed-length feature vector by average pooling
                pooled_features = F.adaptive_avg_pool2d(feature_tensor.unsqueeze(0), (1, 1))
                feature_vector = pooled_features.view(1, -1).cpu().numpy()
                
                # Match feature dimensions to what the model expects
                if feature_vector.shape[1] > input_dim:
                    feature_vector = feature_vector[:, :input_dim]
                elif feature_vector.shape[1] < input_dim:
                    padded = np.zeros((1, input_dim))
                    padded[:, :feature_vector.shape[1]] = feature_vector
                    feature_vector = padded
                
                # Make prediction
                _, confidences = model.predict(feature_vector)
                
                # Apply threshold
                prediction = 1 if confidences[0] >= threshold else 0
                confidence = confidences[0]
                result = "FORGED" if prediction == 1 else "AUTHENTIC"
                
                # Record result
                results[result.lower()].append(image_file)
                
                # Write to results file
                f.write(f"Image: {image_file}\n")
                f.write(f"Result: {result}\n")
                f.write(f"Confidence: {confidence:.4f}\n\n")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                f.write(f"Image: {image_file}\n")
                f.write(f"Result: ERROR - {str(e)}\n\n")
                results['errors'].append(image_file)
    
    # Print summary
    logger.info("\nProcessing complete!")
    logger.info(f"Processed {num_images} images")
    logger.info(f"Authentic images: {len(results['authentic'])}")
    logger.info(f"Forged images: {len(results['forged'])}")
    logger.info(f"Errors: {len(results['errors'])}")
    
    logger.info(f"Results saved to {results_file}")

def localize_mode(args: argparse.Namespace) -> None:
    """Localize forgery in the input image or directory
    
    Args:
        args: Command line arguments
    """
    from modules.feature_extractor import PDyWTCNNDetector
    
    # Initialize detector with models
    detector = PDyWTCNNDetector(
        model_path=args.model_path,
        localization_model_path=args.localization_model_path
    )
    
    # Get threshold
    threshold = args.threshold
    if threshold is None and hasattr(detector.rdlnn_model, 'threshold'):
        threshold = detector.rdlnn_model.threshold
    elif threshold is None:
        threshold = 0.675
    
    logger.info(f"Using classification threshold: {threshold}")
    
    if args.image_path:
        # Process single image
        logger.info(f"Localizing forgery in image: {args.image_path}")
        result = detector.detect(args.image_path)
        
        # Apply threshold
        prediction = 1 if result.get('probability', 0) >= threshold else 0
        
        if prediction == 1:
            # Image detected as forged, perform localization
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_forgery_map.png")
            localization_result = detector.localize(args.image_path, save_path=output_path)
            
            logger.info(f"Localization result saved to {output_path}")
            if localization_result.get('region_proposals', []):
                logger.info(f"Found {len(localization_result['region_proposals'])} suspicious regions.")
        else:
            logger.info("Image appears authentic, no localization performed.")
    
    elif args.input_dir:
        # Process directory of images
        image_files = [f for f in os.listdir(args.input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for image_file in image_files:
            image_path = os.path.join(args.input_dir, image_file)
            logger.info(f"Processing {image_path}")
            
            result = detector.detect(image_path)
            
            # Apply threshold
            prediction = 1 if result.get('probability', 0) >= threshold else 0
            
            if prediction == 1:
                # Image detected as forged, perform localization
                output_path = os.path.join(args.output_dir, f"{os.path.splitext(image_file)[0]}_forgery_map.png")
                localization_result = detector.localize(image_path, save_path=output_path)
                
                logger.info(f"Localization result saved to {output_path}")
                if localization_result.get('region_proposals', []):
                    logger.info(f"Found {len(localization_result['region_proposals'])} suspicious regions.")
            else:
                logger.info(f"{image_file} appears authentic, no localization performed.")
    else:
        logger.error("Error: Please provide either --image_path or --input_dir argument for localization")

def test_single_image(args: argparse.Namespace) -> None:
    """Test the model on a single image using ensemble prediction for robustness"""
    if not args.image_path:
        logger.error("Error: Please provide --image_path argument for single image testing")
        return
    
    if not os.path.exists(args.model_path):
        logger.error(f"Error: Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        logger.error(f"Error: Image not found at {args.image_path}")
        return
    
    # Load the model
    model = RegressionDLNN.load(args.model_path)
    
    # Get the expected feature dimension from the model
    input_dim = model.model[0].in_features
    logger.info(f"Model expects {input_dim} features")
    
    # Check if model has a custom threshold
    threshold = getattr(model, 'threshold', 0.7)  # Default to 0.7 as identified in training
    logger.info(f"Using classification threshold: {threshold}")

    if args.threshold:
        threshold = args.threshold
        logger.info(f"Overriding with command threshold: {threshold}")
    
    # Process the image using ensemble approach
    logger.info(f"Processing single image: {args.image_path}")
    start_time = time.time()
    
    try:
        # Initialize detector
        detector = PDyWTCNNDetector()
        
        # Use the full feature extraction method instead of just wavelet features
        feature_vector = detector.extract_features(args.image_path)
        
        if feature_vector is None:
            logger.error("Error: Failed to extract features")
            return
            
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Apply feature selection if available in the model
        if hasattr(model, 'feature_selector') and model.feature_selector is not None:
            try:
                feature_vector = model.feature_selector(feature_vector)
                logger.info(f"Applied feature selection, new shape: {feature_vector.shape}")
            except Exception as e:
                logger.error(f"Error applying feature selection: {e}")
        
        # Match feature dimensions to what the model expects
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
        
        # Make prediction
        _, confidences = model.predict(feature_vector)
        
        # Apply threshold
        prediction = 1 if confidences[0] >= threshold else 0
        confidence = confidences[0]
        result = "FORGED" if prediction == 1 else "AUTHENTIC"
        
        logger.info(f"\nResult: {result}")
        logger.info(f"Confidence: {confidence:.4f}")
        logger.info(f"Threshold: {threshold}")
        
        # Save the result
        result_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_result.txt")
        with open(result_file, 'w') as f:
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Result: {result}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
            f.write(f"Threshold: {threshold}\n")
        
        logger.info(f"Result saved to {result_file}")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

def analyze_features(args: argparse.Namespace) -> None:
    """Analyze precomputed features to gain insights
    
    Args:
        args: Command line arguments
    """
    import numpy as np
    
    if not os.path.exists(args.features_path):
        logger.error(f"Error: Features file not found at {args.features_path}")
        return
    
    # Load features
    features, labels, paths = load_and_verify_features(args.features_path)
    
    if len(features) == 0:
        logger.error("Error: No valid features found for analysis.")
        return
    
    if len(labels) == 0:
        logger.error("Error: No labels found. Feature analysis requires labeled data.")
        return
    
    logger.info("\nFeature Analysis:")
    logger.info("================")
    
    # Split features by class
    authentic_features = features[labels == 0]
    forged_features = features[labels == 1]
    
    logger.info(f"Authentic samples: {len(authentic_features)}")
    logger.info(f"Forged samples: {len(forged_features)}")
    
    # Compute feature statistics by class
    authentic_mean = np.mean(authentic_features, axis=0)
    forged_mean = np.mean(forged_features, axis=0)
    
    # Find most discriminative features
    feature_diff = np.abs(authentic_mean - forged_mean)
    top_features = np.argsort(-feature_diff)[:10]  # Top 10 features
    
    logger.info("\nTop 10 most discriminative features:")
    for i, feature_idx in enumerate(top_features):
        logger.info(f"{i+1}. Feature {feature_idx}: "
              f"Auth={authentic_mean[feature_idx]:.4f}, "
              f"Forged={forged_mean[feature_idx]:.4f}, "
              f"Diff={feature_diff[feature_idx]:.4f}")
    
    # Plot feature distributions
    plt.figure(figsize=(15, 10))
    
    # Plot mean feature values
    plt.subplot(2, 1, 1)
    plt.plot(authentic_mean, 'b-', alpha=0.7, label='Authentic')
    plt.plot(forged_mean, 'r-', alpha=0.7, label='Forged')
    plt.title('Mean Feature Values by Class')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot feature difference
    plt.subplot(2, 1, 2)
    plt.bar(range(len(feature_diff)), feature_diff)
    plt.title('Absolute Difference in Feature Means')
    plt.xlabel('Feature Index')
    plt.ylabel('Absolute Difference')
    plt.grid(True, alpha=0.3)
    
    # Highlight top discriminative features
    for feature_idx in top_features:
        plt.axvline(x=feature_idx, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    analysis_file = os.path.join(args.output_dir, 'feature_analysis.png')
    plt.savefig(analysis_file)
    logger.info(f"\nFeature analysis plot saved to {analysis_file}")
    
    # PCA visualization if we have enough samples
    if len(features) > 2:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply PCA
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            # Plot PCA
            plt.figure(figsize=(10, 8))
            plt.scatter(features_pca[labels == 0, 0], features_pca[labels == 0, 1], 
                        c='blue', marker='o', alpha=0.7, label='Authentic')
            plt.scatter(features_pca[labels == 1, 0], features_pca[labels == 1, 1], 
                        c='red', marker='x', alpha=0.7, label='Forged')
            plt.title('PCA Visualization of Features')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the PCA plot
            pca_file = os.path.join(args.output_dir, 'pca_visualization.png')
            plt.savefig(pca_file)
            logger.info(f"PCA visualization saved to {pca_file}")
        except ImportError:
            logger.warning("scikit-learn not installed. Skipping PCA visualization.")
        except Exception as e:
            logger.error(f"Error during PCA visualization: {e}", exc_info=args.debug)

if __name__ == "__main__":
    main()