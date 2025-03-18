#!/usr/bin/env python3
"""
Training script for the PDyWT-CNN forgery detection and localization models.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from modules.pdywt_cnn import PDyWTCNNDetector
from modules.utils import logger, setup_logging


def train_detection_model(features_path, output_path, validation_split=0.2, 
                        epochs=30, batch_size=32, learning_rate=0.001):
    """
    Train the forgery detection model using extracted PDyWT features
    
    Args:
        features_path: Path to the features file
        output_path: Path to save the trained model
        validation_split: Fraction of data to use for validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Training history dictionary
    """
    logger.info(f"Loading features from {features_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize detector
    detector = PDyWTCNNDetector()
    
    # Train the model
    history = detector.train_detector(
        features_path=features_path,
        labels=None,  # Assume labels are included in the features file
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Plot training history
    plot_training_history(history, os.path.dirname(output_path))
    
    return history


def train_localization_model(data_dir, annotations_dir, output_path,
                           epochs=30, batch_size=8, learning_rate=0.0005):
    """
    Train the forgery localization model using image data and pixel-level annotations
    
    Args:
        data_dir: Directory containing image files
        annotations_dir: Directory containing annotation masks (binary masks of forged regions)
        output_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Training history dictionary
    """
    logger.info(f"Training localization model with data from {data_dir} and annotations from {annotations_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize detector
    detector = PDyWTCNNDetector()
    
    # Train the model
    history = detector.train_localizer(
        data_dir=data_dir,
        annotations_dir=annotations_dir,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Plot training history
    plot_localization_history(history, os.path.dirname(output_path))
    
    return history


def plot_training_history(history, output_dir):
    """Plot and save training history metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_training_history.png'))
    plt.close()


def plot_localization_history(history, output_dir):
    """Plot and save localization training history metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot Dice score
    plt.subplot(1, 2, 2)
    plt.plot(history['dice_score'], label='Dice Score')
    plt.title('Training Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'localization_training_history.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train PDyWT-CNN forgery detection and localization models")
    
    # Training mode selection
    parser.add_argument('--mode', choices=['detection', 'localization', 'both'], default='both',
                      help='Training mode: detection, localization, or both')
    
    # Detection model training parameters
    parser.add_argument('--features', type=str, default='data/features/pdywt_features.npz',
                      help='Path to features file for detection model training')
    
    parser.add_argument('--detection_model_output', type=str, default='data/models/pdywt_detector.pth',
                      help='Path to save detection model')
    
    parser.add_argument('--detection_epochs', type=int, default=30,
                      help='Number of epochs for detection model training')
    
    parser.add_argument('--detection_batch_size', type=int, default=32,
                      help='Batch size for detection model training')
    
    parser.add_argument('--detection_lr', type=float, default=0.001,
                      help='Learning rate for detection model training')
    
    # Localization model training parameters
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Directory containing image files for localization training')
    
    parser.add_argument('--annotations_dir', type=str, default=None,
                      help='Directory containing annotation masks for localization training')
    
    parser.add_argument('--localization_model_output', type=str, default='data/models/pdywt_localizer.pth',
                      help='Path to save localization model')
    
    parser.add_argument('--localization_epochs', type=int, default=30,
                      help='Number of epochs for localization model training')
    
    parser.add_argument('--localization_batch_size', type=int, default=8,
                      help='Batch size for localization model training')
    
    parser.add_argument('--localization_lr', type=float, default=0.0005,
                      help='Learning rate for localization model training')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(os.path.dirname(args.detection_model_output))
    
    # Train models based on mode
    if args.mode in ['detection', 'both']:
        if not os.path.exists(args.features):
            logger.error(f"Features file not found: {args.features}")
        else:
            logger.info(f"Training detection model using features from {args.features}")
            train_detection_model(
                features_path=args.features,
                output_path=args.detection_model_output,
                epochs=args.detection_epochs,
                batch_size=args.detection_batch_size,
                learning_rate=args.detection_lr
            )
    
    if args.mode in ['localization', 'both']:
        if not args.data_dir or not args.annotations_dir:
            logger.error("Both --data_dir and --annotations_dir are required for localization training")
        elif not os.path.isdir(args.data_dir) or not os.path.isdir(args.annotations_dir):
            logger.error(f"Data or annotations directory not found")
        else:
            logger.info(f"Training localization model")
            train_localization_model(
                data_dir=args.data_dir,
                annotations_dir=args.annotations_dir,
                output_path=args.localization_model_output,
                epochs=args.localization_epochs,
                batch_size=args.localization_batch_size,
                learning_rate=args.localization_lr
            )
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()