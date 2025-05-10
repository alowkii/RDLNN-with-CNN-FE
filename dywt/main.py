import argparse
import os
import numpy as np
from utils import load_dataset, load_model, save_model
from train import train_model, save_report
from test import detect_forgery

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Dyadic Wavelet Transform Image Forgery Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--authentic_dir', type=str, required=True,
                        help='Directory containing authentic images')
    train_parser.add_argument('--forged_dir', type=str, required=True,
                        help='Directory containing forged images')
    train_parser.add_argument('--model_path', type=str, default="dyadic_forgery_model.pkl",
                        help='Path to save the trained model')
    train_parser.add_argument('--report_dir', type=str, default="training_reports",
                        help='Directory to save training reports')
    train_parser.add_argument('--decomp_level', type=int, default=3,
                        help='Level of wavelet decomposition')
    
    # Test subcommand
    test_parser = subparsers.add_parser('test', help='Test the model on a single image')
    test_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    test_parser.add_argument('--test_image', type=str, required=True,
                        help='Path to test image for forgery detection')
    test_parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for forgery classification')
    
    # Batch test subcommand
    batch_parser = subparsers.add_parser('batch_test', help='Test the model on a directory of images')
    batch_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    batch_parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    batch_parser.add_argument('--output_csv', type=str, default="results.csv",
                        help='Path to save the results CSV')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Loading and extracting features from dataset...")
        X, y = load_dataset(args.authentic_dir, args.forged_dir, decomp_level=args.decomp_level)
        
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
        result, probability = detect_forgery(model, args.test_image, threshold=args.threshold)
        if result is not None:
            status = "FORGED" if result == 1 else "AUTHENTIC"
            print(f"Image {args.test_image} is {status.lower()} (confidence: {probability:.2f})")
            print(f"Prediction: {status}")
            
    elif args.command == 'batch_test':
        # Load the model
        model = load_model(args.model_path)
        
        # Check if test directory exists
        if not os.path.isdir(args.test_dir):
            print(f"Error: The specified test directory '{args.test_dir}' does not exist.")
            return
            
        # Process all images in the directory
        results = []
        for img_name in os.listdir(args.test_dir):
            img_path = os.path.join(args.test_dir, img_name)
            if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                result, probability = detect_forgery(model, img_path)
                if result is not None:
                    results.append((img_name, result, probability))
                    status = "FORGED" if result == 1 else "AUTHENTIC"
                    print(f"{img_name}: {status} (confidence: {probability:.2f})")
        
        # Save results to CSV
        if results:
            import csv
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'Prediction', 'Confidence'])
                for img_name, result, probability in results:
                    status = "FORGED" if result == 1 else "AUTHENTIC"
                    writer.writerow([img_name, status, f"{probability:.4f}"])
            print(f"Results saved to {args.output_csv}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()