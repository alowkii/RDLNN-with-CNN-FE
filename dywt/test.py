import argparse
import os
import numpy as np
import csv
from .utils import extract_dyadic_wavelet_features, load_model

def detect_forgery(model, image_path, threshold=0.5):
    """
    Detect if an image is forged using the trained model.
    
    Args:
        model: Trained classifier or model dictionary
        image_path: Path to the image to analyze
        threshold: Probability threshold for forgery classification
        
    Returns:
        Prediction (0 for authentic, 1 for forged) and probability
    """
    features = extract_dyadic_wavelet_features(image_path)
    if features is None:
        print(f"Could not process image: {image_path}")
        return None, None
    
    # Check if model is a dictionary containing model and scaler
    if isinstance(model, dict) and 'model' in model and 'scaler' in model:
        # Scale features
        features_scaled = model['scaler'].transform([features])
        
        # Get prediction and probability
        prediction_prob = model['model'].predict_proba(features_scaled)[0, 1]
        prediction = 1 if prediction_prob >= threshold else 0
        
        return prediction, prediction_prob
    else:
        # Legacy support for older models without scaler
        prediction = model.predict([features])[0]
        try:
            prediction_prob = model.predict_proba([features])[0, 1]
        except:
            prediction_prob = float(prediction)
            
        return prediction, prediction_prob

def batch_test(model, test_dir, output_csv="results.csv", threshold=0.5):
    """
    Test the model on a directory of images.
    
    Args:
        model: Trained classifier
        test_dir: Directory containing test images
        output_csv: Path to save the results CSV
        threshold: Probability threshold for forgery classification
    """
    results = []
    
    # Process all images in the directory
    for img_name in os.listdir(test_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(test_dir, img_name)
            result, probability = detect_forgery(model, img_path, threshold)
            if result is not None:
                results.append((img_name, result, probability))
                status = "FORGED" if result == 1 else "AUTHENTIC"
                print(f"{img_name}: {status} (confidence: {probability:.2f})")
    
    # Save results to CSV
    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Prediction', 'Confidence'])
            for img_name, result, probability in results:
                status = "FORGED" if result == 1 else "AUTHENTIC"
                writer.writerow([img_name, status, f"{probability:.4f}"])
        print(f"Results saved to {output_csv}")
    else:
        print("No valid images found for testing.")

def visualize_results(test_dir, results_csv, output_dir="test_visualizations"):
    """
    Create visual representations of test results.
    
    Args:
        test_dir: Directory containing test images
        results_csv: CSV file with test results
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import cv2
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load results from CSV
    results = {}
    with open(results_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            img_name, prediction, confidence = row
            results[img_name] = (prediction, float(confidence))
    
    # Process images
    for img_name, (prediction, confidence) in results.items():
        img_path = os.path.join(test_dir, img_name)
        if os.path.exists(img_path):
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create figure with image and prediction
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img_rgb)
            
            # Add prediction text with background box
            status = "FORGED" if prediction == "FORGED" else "AUTHENTIC"
            color = 'red' if status == "FORGED" else 'green'
            
            # Add title with prediction
            ax.set_title(f"Prediction: {status} (Confidence: {confidence:.2f})", 
                         color=color, fontsize=14)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Save visualization
            output_path = os.path.join(output_dir, f"result_{img_name}")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            print(f"Visualization for {img_name} saved to {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Dyadic Wavelet Transform Image Forgery Detection Model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single test subcommand
    single_parser = subparsers.add_parser('single', help='Test the model on a single image')
    single_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    single_parser.add_argument('--test_image', type=str, required=True,
                        help='Path to test image for forgery detection')
    single_parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for forgery classification')
    
    # Batch test subcommand
    batch_parser = subparsers.add_parser('batch', help='Test the model on a directory of images')
    batch_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    batch_parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    batch_parser.add_argument('--output_csv', type=str, default="results.csv",
                        help='Path to save the results CSV')
    batch_parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for forgery classification')
    batch_parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of results')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        # Check if test image exists
        if not os.path.isfile(args.test_image):
            print(f"Error: The specified test image '{args.test_image}' does not exist.")
            return
        
        # Load the model
        model = load_model(args.model_path)
        
        # Test the image
        result, probability = detect_forgery(model, args.test_image, threshold=args.threshold)
        if result is not None:
            status = "FORGED" if result == 1 else "AUTHENTIC"
            print(f"Image {args.test_image} is {status.lower()}")
            print(f"Prediction: {status} (confidence: {probability:.2f})")
    
    elif args.command == 'batch':
        # Check if test directory exists
        if not os.path.isdir(args.test_dir):
            print(f"Error: The specified test directory '{args.test_dir}' does not exist.")
            return
        
        # Load the model
        model = load_model(args.model_path)
        
        # Batch test
        batch_test(model, args.test_dir, args.output_csv, args.threshold)
        
        # Create visualizations if requested
        if args.visualize and os.path.exists(args.output_csv):
            visualize_results(args.test_dir, args.output_csv)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()