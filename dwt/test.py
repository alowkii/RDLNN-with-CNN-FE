# test.py
import argparse
import os
from utils import extract_dwt_features, load_model

def detect_forgery(model, image_path):
    """
    Detect if an image is forged using the trained model.
    
    Args:
        model: Trained classifier
        image_path: Path to the image to analyze
        
    Returns:
        Prediction (0 for authentic, 1 for forged)
    """
    features = extract_dwt_features(image_path)
    if features is None:
        print(f"Could not process image: {image_path}")
        return None
    
    prediction = model.predict([features])[0]
    return prediction

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test DWT Image Forgery Detection Model on a single image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--test_image', type=str, required=True,
                        help='Path to test image for forgery detection')
    
    args = parser.parse_args()
    
    # Check if test image exists
    if not os.path.isfile(args.test_image):
        print(f"Error: The specified test image '{args.test_image}' does not exist.")
        return
    
    # Load the model
    model = load_model(args.model_path)
    
    # Test the image
    result = detect_forgery(model, args.test_image)
    if result is not None:
        print(f"Image {args.test_image} is {'forged' if result == 1 else 'authentic'}")
        print(f"Prediction: {"authentic".upper() if result == 0 else "forged".upper()}")

if __name__ == "__main__":
    main()