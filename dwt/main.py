# dwt_forgery_detection.py
import argparse
import os
import numpy as np
from utils import load_dataset, load_model, save_model
from train import train_model, save_report
from test import detect_forgery

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='DWT Image Forgery Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--authentic_dir', type=str, required=True,
                        help='Directory containing authentic images')
    train_parser.add_argument('--forged_dir', type=str, required=True,
                        help='Directory containing forged images')
    train_parser.add_argument('--model_path', type=str, default="dwt_forgery_model.pkl",
                        help='Path to save the trained model')
    train_parser.add_argument('--report_dir', type=str, default="training_reports",
                        help='Directory to save training reports')
    
    # Test subcommand
    test_parser = subparsers.add_parser('test', help='Test the model on a single image')
    test_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    test_parser.add_argument('--test_image', type=str, required=True,
                        help='Path to test image for forgery detection')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Loading and extracting features from dataset...")
        X, y = load_dataset(args.authentic_dir, args.forged_dir)
        
        print(f"Dataset loaded: {len(X)} images, {np.sum(y)} forged, {len(X) - np.sum(y)} authentic")
        
        print("Training model...")
        model, test_data, performance = train_model(X, y)
        
        # Save the model
        save_model(model, args.model_path)
        
        # Save the report
        report, conf_matrix, accuracy = performance
        save_report(report, conf_matrix, accuracy, args.report_dir)
        
    elif args.command == 'test':
        # Load the model
        model = load_model(args.model_path)
        
        # Test the image
        result = detect_forgery(model, args.test_image)
        if result is not None:
            print(f"Image {args.test_image} is {'forged' if result == 1 else 'authentic'}")
            print(f"Prediction: {"authentic".upper() if result == 0 else "forged".upper()}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()