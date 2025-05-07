import numpy as np
import cv2
import pywt
import os
import pickle

def extract_dwt_features(image_path):
    """
    Extract Discrete Wavelet Transform features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Feature vector containing DWT statistics
    """
    # Read image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply 2D DWT using Haar wavelet
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # Extract statistical features from each sub-band
    features = []
    
    # Process each coefficient matrix
    for coef in [cA, cH, cV, cD]:
        # Calculate statistical features
        mean = np.mean(coef)
        std = np.std(coef)
        entropy = np.sum(-coef * np.log2(np.abs(coef) + 1e-10))
        energy = np.sum(coef ** 2)
        
        # Add features to our vector
        features.extend([mean, std, entropy, energy])
    
    return features

def load_dataset(authentic_dir, forged_dir):
    """
    Load images from authentic and forged directories and extract features.
    
    Args:
        authentic_dir: Directory containing authentic images
        forged_dir: Directory containing forged images
        
    Returns:
        X: Feature matrix
        y: Target labels (0 for authentic, 1 for forged)
    """
    X = []
    y = []
    
    # Process authentic images
    for img_name in os.listdir(authentic_dir):
        img_path = os.path.join(authentic_dir, img_name)
        features = extract_dwt_features(img_path)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 for authentic
    
    # Process forged images
    for img_name in os.listdir(forged_dir):
        img_path = os.path.join(forged_dir, img_name)
        features = extract_dwt_features(img_path)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 for forged
    
    return np.array(X), np.array(y)

def save_model(model, filename="dwt_forgery_model.pkl"):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained classifier model
        filename: Path to save the model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename="dwt_forgery_model.pkl"):
    """
    Load a trained model from a file.
    
    Args:
        filename: Path to the saved model
    
    Returns:
        Trained model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model# dwt_utils.py
import numpy as np
import cv2
import pywt
import os
import pickle

def extract_dwt_features(image_path):
    """
    Extract Discrete Wavelet Transform features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Feature vector containing DWT statistics
    """
    # Read image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply 2D DWT using Haar wavelet
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # Extract statistical features from each sub-band
    features = []
    
    # Process each coefficient matrix
    for coef in [cA, cH, cV, cD]:
        # Calculate statistical features
        mean = np.mean(coef)
        std = np.std(coef)
        entropy = np.sum(-coef * np.log2(np.abs(coef) + 1e-10))
        energy = np.sum(coef ** 2)
        
        # Add features to our vector
        features.extend([mean, std, entropy, energy])
    
    return features

def load_dataset(authentic_dir, forged_dir):
    """
    Load images from authentic and forged directories and extract features.
    
    Args:
        authentic_dir: Directory containing authentic images
        forged_dir: Directory containing forged images
        
    Returns:
        X: Feature matrix
        y: Target labels (0 for authentic, 1 for forged)
    """
    X = []
    y = []
    
    # Process authentic images
    for img_name in os.listdir(authentic_dir):
        img_path = os.path.join(authentic_dir, img_name)
        features = extract_dwt_features(img_path)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 for authentic
    
    # Process forged images
    for img_name in os.listdir(forged_dir):
        img_path = os.path.join(forged_dir, img_name)
        features = extract_dwt_features(img_path)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 for forged
    
    return np.array(X), np.array(y)

def save_model(model, filename="dwt_forgery_model.pkl"):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained classifier model
        filename: Path to save the model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename="dwt_forgery_model.pkl"):
    """
    Load a trained model from a file.
    
    Args:
        filename: Path to the saved model
    
    Returns:
        Trained model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model