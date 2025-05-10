import numpy as np
import cv2
import pywt
import os
import pickle
from skimage.feature import graycomatrix, graycoprops

def extract_dyadic_wavelet_features(image_path, decomp_level=3):
    """
    Extract Dyadic Wavelet Transform features from an image.
    
    Args:
        image_path: Path to the image file
        decomp_level: Level of wavelet decomposition
        
    Returns:
        Feature vector containing DWT statistics
    """
    # Read image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply multilevel 2D DWT using Daubechies wavelet
    coeffs = pywt.wavedec2(gray, 'db4', level=decomp_level)
    
    # Extract statistical features from each sub-band
    features = []
    
    # Process approximation coefficient matrix (cA)
    cA = coeffs[0]
    features.extend(extract_statistical_features(cA))
    
    # Process detail coefficient matrices (cH, cV, cD) for each level
    for detail_coeffs in coeffs[1:]:
        cH, cV, cD = detail_coeffs
        features.extend(extract_statistical_features(cH))
        features.extend(extract_statistical_features(cV))
        features.extend(extract_statistical_features(cD))
    
    # Add texture features from GLCM on the approximation coefficients
    glcm_features = extract_glcm_features(cA)
    features.extend(glcm_features)
    
    return features

def extract_statistical_features(coef):
    """
    Extract statistical features from a coefficient matrix.
    
    Args:
        coef: Wavelet coefficient matrix
        
    Returns:
        List of statistical features
    """
    # Calculate statistical features
    mean = np.mean(coef)
    std = np.std(coef)
    skewness = np.mean(((coef - mean) / (std + 1e-10)) ** 3) if std > 0 else 0
    kurtosis = np.mean(((coef - mean) / (std + 1e-10)) ** 4) - 3 if std > 0 else 0
    entropy = -np.sum((np.abs(coef) + 1e-10) * np.log2(np.abs(coef) + 1e-10))
    energy = np.sum(coef ** 2)
    
    return [mean, std, skewness, kurtosis, entropy, energy]

def extract_glcm_features(coef):
    """
    Extract Gray Level Co-occurrence Matrix (GLCM) features from coefficient matrix.
    
    Args:
        coef: Wavelet coefficient matrix
        
    Returns:
        List of GLCM features
    """
    # Normalize coefficients to 0-255 range for GLCM
    coef_norm = np.uint8(255 * (coef - np.min(coef)) / (np.max(coef) - np.min(coef) + 1e-10))
    
    # Generate GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(coef_norm, distances, angles, 256, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Take mean values across angles for each property
    contrast_mean = np.mean(contrast)
    dissimilarity_mean = np.mean(dissimilarity)
    homogeneity_mean = np.mean(homogeneity)
    energy_mean = np.mean(energy)
    correlation_mean = np.mean(correlation)
    
    return [contrast_mean, dissimilarity_mean, homogeneity_mean, energy_mean, correlation_mean]

def load_dataset(authentic_dir, forged_dir, decomp_level=3):
    """
    Load images from authentic and forged directories and extract features.
    
    Args:
        authentic_dir: Directory containing authentic images
        forged_dir: Directory containing forged images
        decomp_level: Level of wavelet decomposition
        
    Returns:
        X: Feature matrix
        y: Target labels (0 for authentic, 1 for forged)
    """
    X = []
    y = []
    
    # Process authentic images
    for img_name in os.listdir(authentic_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(authentic_dir, img_name)
            features = extract_dyadic_wavelet_features(img_path, decomp_level)
            if features is not None:
                X.append(features)
                y.append(0)  # 0 for authentic
    
    # Process forged images
    for img_name in os.listdir(forged_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(forged_dir, img_name)
            features = extract_dyadic_wavelet_features(img_path, decomp_level)
            if features is not None:
                X.append(features)
                y.append(1)  # 1 for forged
    
    return np.array(X), np.array(y)

def save_model(model, filename="dyadic_forgery_model.pkl"):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained classifier model
        filename: Path to save the model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename="dyadic_forgery_model.pkl"):
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