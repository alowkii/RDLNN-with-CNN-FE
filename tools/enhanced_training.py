#!/usr/bin/env python3
"""
Enhanced training script for forgery detection with ensemble methods
"""

import os
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from your modules
from modules.rdlnn import RegressionDLNN
from modules.utils import setup_logging, logger

def load_and_verify_features(features_path):
    """
    Load and verify feature file
    
    Args:
        features_path: Path to the features file
        
    Returns:
        Tuple of (features, labels, paths)
    """
    logger.info(f"Loading features from {features_path}")
    
    try:
        data = np.load(features_path, allow_pickle=True)
        features = data['features']
        labels = data.get('labels', np.array([]))
        paths = data.get('paths', np.array([]))
        
        # Log basic info
        logger.info(f"Loaded {len(features)} feature vectors")
        if len(labels) > 0:
            logger.info(f"Class distribution: {np.bincount(labels.astype(int))}")
        
        return features, labels, paths
        
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return np.array([]), np.array([]), []

def train_enhanced_model(features_path, model_path, output_dir, ensemble_size=3, 
                        epochs=30, learning_rate=0.001, batch_size=32,
                        use_deep_architecture=True):
    """
    Train an enhanced model with ensemble methods
    
    Args:
        features_path: Path to features file
        model_path: Path to save model
        output_dir: Directory for output files
        ensemble_size: Number of models in ensemble
        epochs: Training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        use_deep_architecture: Whether to use the enhanced architecture
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    logger.info("Starting enhanced training with ensemble methods")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return None
    
    # Split data into train/validation/test sets (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.4, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Training: {len(X_train)} samples | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # Train ensemble of models
    ensemble_models = []
    for i in range(ensemble_size):
        logger.info(f"Training model {i+1}/{ensemble_size}")
        
        # Create model with architecture based on flag
        architecture = 'deep' if use_deep_architecture else 'standard'
        model = RegressionDLNN(features.shape[1], architecture=architecture)
        
        # Create bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]
        
        # Train model
        history = model.fit(
            X_bootstrap, y_bootstrap,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_split=0.2,
            early_stopping=5,
            use_fp16=True,
            force_class_balance=True
        )
        
        # Save individual model
        model_file = os.path.splitext(model_path)[0] + f"_ensemble_{i}.pth"
        model.save(model_file)
        logger.info(f"Model {i+1} saved to {model_file}")
        
        ensemble_models.append(model)
    
    # Evaluate ensemble on test set
    ensemble_preds = []
    ensemble_probs = []
    
    for i, model in enumerate(ensemble_models):
        # Make predictions
        preds, probs = model.predict(X_test)
        ensemble_preds.append(preds)
        ensemble_probs.append(probs)
        
        # Calculate individual model metrics
        accuracy = np.mean(preds == y_test)
        logger.info(f"Model {i+1} test accuracy: {accuracy:.4f}")
    
    # Combine predictions by majority vote
    ensemble_preds = np.array(ensemble_preds)
    final_preds = np.round(np.mean(ensemble_preds, axis=0)).astype(int)
    
    # Combine probabilities by averaging
    ensemble_probs = np.array(ensemble_probs)
    final_probs = np.mean(ensemble_probs, axis=0)
    
    # Calculate ensemble metrics
    accuracy = np.mean(final_preds == y_test)
    
    # Calculate confusion matrix
    tp = np.sum((final_preds == 1) & (y_test == 1))
    tn = np.sum((final_preds == 0) & (y_test == 0))
    fp = np.sum((final_preds == 1) & (y_test == 0))
    fn = np.sum((final_preds == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info("\nEnsemble Performance:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"True Positive: {tp}")
    logger.info(f"True Negative: {tn}")
    logger.info(f"False Positive: {fp}")
    logger.info(f"False Negative: {fn}")
    
    # Save a reference model that includes threshold and other settings
    reference_model = ensemble_models[0]  # Use first model as reference
    
    # Find optimal threshold on validation set
    val_probs = []
    for model in ensemble_models:
        _, probs = model.predict(X_val)
        val_probs.append(probs)
    
    # Average probabilities from all models
    val_probs = np.mean(np.array(val_probs), axis=0)
    
    # Try different thresholds
    best_f1 = 0
    best_threshold = 0.5
    
    thresholds = np.arange(0.3, 0.8, 0.05)
    for thresh in thresholds:
        val_preds = (val_probs >= thresh).astype(int)
        
        # Calculate metrics
        tp_val = np.sum((val_preds == 1) & (y_val == 1))
        fp_val = np.sum((val_preds == 1) & (y_val == 0))
        fn_val = np.sum((val_preds == 0) & (y_val == 1))
        
        precision_val = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
        recall_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        if f1_val > best_f1:
            best_f1 = f1_val
            best_threshold = thresh
    
    logger.info(f"\nOptimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Set optimal threshold in reference model and save
    reference_model.threshold = best_threshold
    reference_model.save(model_path)
    
    logger.info(f"Reference model saved with threshold {best_threshold} to {model_path}")
    logger.info("Enhanced training complete")
    
    return ensemble_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced training for forgery detection")
    parser.add_argument('--features_path', type=str, required=True,
                      help='Path to features file')
    parser.add_argument('--model_path', type=str, default='data/models/enhanced_model.pth',
                      help='Path to save the reference model')
    parser.add_argument('--output_dir', type=str, default='data/results',
                      help='Directory to save results')
    parser.add_argument('--ensemble_size', type=int, default=3,
                      help='Number of models in ensemble')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--use_deep_architecture', action='store_true',
                      help='Use enhanced deep architecture')
    
    args = parser.parse_args()
    
    train_enhanced_model(
        args.features_path,
        args.model_path,
        args.output_dir,
        args.ensemble_size,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.use_deep_architecture
    )