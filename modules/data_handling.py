"""
Data handling functions for the forgery detection system
"""

import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
import torch.nn.functional as F

from modules.preprocessing import preprocess_image
from modules.feature_extractor import PDyWTCNNDetector
from modules.utils import logger, clean_cuda_memory

class RobustFeatureSelector:
    def __init__(self, variance_selector, indices, expected_features):
        self.variance_selector = variance_selector
        self.indices = indices
        self.expected_features = expected_features
        
    def __call__(self, X):
        # Check if dimensions match what's expected
        if X.shape[1] != self.expected_features:
            logger.warning(f"Feature dimension mismatch: got {X.shape[1]}, expected {self.expected_features}")
            
            # Handle dimension mismatch
            if X.shape[1] > self.expected_features:
                # If too many features, truncate
                logger.info(f"Truncating features from {X.shape[1]} to {self.expected_features}")
                X = X[:, :self.expected_features]
            else:
                # If too few features, pad with zeros
                logger.info(f"Padding features from {X.shape[1]} to {self.expected_features}")
                padded = np.zeros((X.shape[0], self.expected_features))
                padded[:, :X.shape[1]] = X
                X = padded
        
        # Apply variance selector and feature selection with error handling
        try:
            # Apply variance selector
            X_var = self.variance_selector.transform(X)
            # Then select the chosen features
            return X_var[:, self.indices]
        except Exception as e:
            logger.error(f"Error applying feature selection, using original features: {e}")
            # In case of error, try to return appropriately sized features
            if hasattr(self.variance_selector, 'n_features_'):
                # If we know the expected output size, pad or truncate
                output_size = len(self.indices)
                if X.shape[1] > output_size:
                    return X[:, :output_size]
                else:
                    padded = np.zeros((X.shape[0], output_size))
                    padded[:, :X.shape[1]] = X
                    return padded
            else:
                # Otherwise just return the input
                return X

def precompute_features(directory: str, label=None, batch_size=16, num_workers=4, use_fp16=True, save_path=None) -> Dict[str, Any]:
    """
    Precompute and optionally save feature vectors for all images in a directory
    
    Args:
        directory: Directory containing images
        label: Label to assign to all images (0=authentic, 1=forged, None=unknown)
        batch_size: Batch size for processing
        num_workers: Number of worker threads
        use_fp16: Whether to use half precision
        save_path: Path to save feature vectors (optional)
        
    Returns:
        Dictionary with image paths, features, and labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Precomputing features using {device}")
    
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    num_images = len(image_files)
    logger.info(f"Found {num_images} images in {directory}")
    
    # Initialize detector for feature extraction
    detector = PDyWTCNNDetector()
    
    # Allocate results
    results = {
        'paths': [],
        'features': [],
        'labels': []
    }
    
    # Process images in batches with a progress bar
    with tqdm(total=num_images, desc="Extracting features", unit="img") as pbar:
        for batch_idx in range(0, num_images, batch_size):
            batch_end = min(batch_idx + batch_size, num_images)
            batch_files = image_files[batch_idx:batch_end]
            batch_paths = [os.path.join(directory, f) for f in batch_files]
            
            # Process each image in the batch
            batch_features = []
            valid_paths = []

            for img_path in batch_paths:
                try:
                    # Extract all features using the detector's extract_features method
                    # which combines wavelet, ELA, and noise features consistently
                    feature_vector = detector.extract_features(img_path)
                    
                    if feature_vector is not None:
                        batch_features.append(feature_vector)
                        valid_paths.append(img_path)
                    else:
                        logger.warning(f"Failed to extract features from {img_path}")
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
            
            if batch_features:
                # Stack features into a batch
                features_array = np.vstack(batch_features)
                
                # Add to results
                results['paths'].extend(valid_paths)
                results['features'].append(features_array)
                
                # Add labels if provided
                if label is not None:
                    results['labels'].extend([label] * len(valid_paths))
            
            # Update progress
            pbar.update(len(batch_paths))
            
            # Clean up GPU memory
            clean_cuda_memory()
    
    # Combine all features
    if results['features']:
        results['features'] = np.vstack(results['features'])
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                paths=results['paths'],
                features=results['features'],
                labels=results.get('labels', [])
            )
            logger.info(f"Saved {len(results['paths'])} feature vectors to {save_path}")
    
    return results

def _process_image_batch(image_paths: List[str], device: torch.device, num_workers: int, 
                        batch_size: int, use_fp16: bool) -> Optional[torch.Tensor]:
    """
    Process a batch of images and extract features
    
    Args:
        image_paths: List of paths to image files
        device: Computation device
        num_workers: Number of worker threads
        batch_size: Batch size
        use_fp16: Whether to use half precision
        
    Returns:
        Tensor of feature vectors for the batch or None if processing failed
    """
    try:
        # Initialize detector
        detector = PDyWTCNNDetector()
        
        # Process batch
        batch_features = []
        valid_paths = []
        
        # Process each image
        for img_path in image_paths:
            try:
                # Preprocess image
                ycbcr_tensor = detector.preprocess_image(img_path)
                if ycbcr_tensor is None:
                    continue
                    
                # Extract wavelet features
                feature_tensor = detector.extract_wavelet_features(ycbcr_tensor)
                
                # Get fixed-length feature vector
                pooled_features = F.adaptive_avg_pool2d(feature_tensor.unsqueeze(0), (1, 1))
                feature_vector = pooled_features.view(-1).cpu().numpy()
                
                batch_features.append(feature_vector)
                valid_paths.append(img_path)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        if not batch_features:
            return None
            
        # Stack into numpy array and convert to tensor
        features_array = np.vstack(batch_features)
        return torch.tensor(features_array, device=device)
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return None

def load_and_verify_features(features_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and verify precomputed features, ensuring features and labels match
    
    Args:
        features_path: Path to the features file
        
    Returns:
        Tuple of (features, labels, paths)
    """
    logger.info(f"Loading precomputed features from {features_path}...")
    
    try:
        data = np.load(features_path, allow_pickle=True)
        features = data['features']
        labels = data.get('labels', np.array([]))
        paths = data.get('paths', np.array([]))
        
        logger.info(f"Original dataset - Features: {features.shape}, Labels: {len(labels)}, Paths: {len(paths)}")
        
        # Check the number of paths, features, and labels
        min_length = min(len(features), len(labels)) if len(labels) > 0 else len(features)
        
        # Trim to make everything consistent
        features = features[:min_length]
        
        if len(labels) > 0:
            labels = np.array(labels[:min_length])
        
        if len(paths) > 0:
            paths = paths[:min_length]
        
        # Filter out NaN values
        valid_indices = []
        for i in range(len(features)):
            # Check if feature is valid (no NaN values)
            if not np.isnan(features[i]).any():
                if len(labels) == 0 or not np.isnan(labels[i] if np.isscalar(labels[i]) else labels[i].any()):
                    valid_indices.append(i)
        
        # Apply filtering
        filtered_features = features[valid_indices]
        filtered_labels = labels[valid_indices] if len(labels) > 0 else []
        filtered_paths = [paths[i] for i in valid_indices] if len(paths) > 0 else []
        
        logger.info(f"Filtered dataset - Features: {filtered_features.shape}")
        
        # Check class distribution if we have labels
        if len(filtered_labels) > 0:
            classes, counts = np.unique(filtered_labels, return_counts=True)
            logger.info("Class distribution:")
            for cls, count in zip(classes, counts):
                logger.info(f"  Class {cls}: {count} samples ({count/len(filtered_labels)*100:.2f}%)")
        
        # Check feature statistics
        logger.info(f"Feature statistics:")
        logger.info(f"  Mean: {np.mean(filtered_features):.4f}")
        logger.info(f"  Std: {np.std(filtered_features):.4f}")
        logger.info(f"  Min: {np.min(filtered_features):.4f}")
        logger.info(f"  Max: {np.max(filtered_features):.4f}")
        
        # Check for NaN values
        nan_count = np.isnan(filtered_features).sum()
        if nan_count > 0:
            logger.warning(f"NaN values detected: {nan_count}")
        else:
            logger.info("No NaN values detected")
        
        return filtered_features, filtered_labels, filtered_paths
        
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return np.array([]), np.array([]), []

def create_training_validation_split(features: np.ndarray, labels: np.ndarray, 
                                     validation_split: float = 0.2, random_seed: int = 42) -> Tuple:
    """
    Create a training-validation split of the data
    
    Args:
        features: Feature vectors
        labels: Labels
        validation_split: Fraction to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_features, val_features, train_labels, val_labels)
    """
    from sklearn.model_selection import train_test_split
    
    try:
        # Use sklearn's train_test_split for stratified splitting
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, 
            test_size=validation_split, 
            random_state=random_seed, 
            stratify=labels
        )
        
        # Log split info
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        if len(y_train) > 0:
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            logger.info("Training class distribution:")
            for cls, count in zip(train_classes, train_counts):
                logger.info(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
                
        if len(y_val) > 0:
            val_classes, val_counts = np.unique(y_val, return_counts=True)
            logger.info("Validation class distribution:")
            for cls, count in zip(val_classes, val_counts):
                logger.info(f"  Class {cls}: {count} samples ({count/len(y_val)*100:.2f}%)")
        
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        logger.error(f"Error creating train-validation split: {e}")
        # Create an emergency split if there's an error
        n = len(features)
        val_size = int(n * validation_split)
        train_size = n - val_size
        return features[:train_size], features[train_size:], labels[:train_size], labels[train_size:]

def get_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights inversely proportional to class frequencies
    
    Args:
        labels: Class labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    unique_classes = np.unique(labels)
    class_counts = np.bincount(labels.flatten().astype(int))
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequencies
    class_weights = {}
    for i, count in enumerate(class_counts):
        if count > 0:  # Avoid division by zero
            class_weights[i] = total_samples / (len(unique_classes) * count)
    
    return class_weights

def get_balanced_sampler(labels: np.ndarray) -> torch.utils.data.sampler.WeightedRandomSampler:
    """
    Create a weighted sampler for balanced batches
    
    Args:
        labels: Class labels
        
    Returns:
        WeightedRandomSampler instance
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Get class counts
    class_count = np.bincount(labels.flatten().astype(int))
    
    # Compute weights for each sample
    class_weights = 1. / class_count
    weights = torch.tensor([class_weights[int(t)] for t in labels.flatten()], dtype=torch.float32)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler

def create_dataloaders(X_train, X_val, y_train, y_val, batch_size=32, num_workers=4, use_balanced_sampling=True):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of worker threads
        use_balanced_sampling: Whether to use balanced sampling for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Calculate optimal number of workers
    num_workers = min(num_workers, os.cpu_count() or 4)
    
    # Create data loaders
    if use_balanced_sampling and np.bincount(y_train.flatten().astype(int))[0] != np.bincount(y_train.flatten().astype(int))[1]:
        logger.info("Using weighted sampler for balanced batches")
        sampler = get_balanced_sampler(y_train)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

class FeatureSelector:
            def __init__(self, variance_selector, indices):
                self.variance_selector = variance_selector
                self.indices = indices
                
            def __call__(self, X):
                X_var = self.variance_selector.transform(X)
                return X_var[:, self.indices]

class RobustFeatureSelector:
    def __init__(self, variance_selector, indices, expected_features):
        self.variance_selector = variance_selector
        self.indices = indices
        self.expected_features = expected_features
        
    def __call__(self, X):
        # Check if dimensions match what's expected
        if X.shape[1] != self.expected_features:
            logger.warning(f"Feature dimension mismatch: got {X.shape[1]}, expected {self.expected_features}")
            if X.shape[1] > self.expected_features:
                # Truncate to expected dimension
                X = X[:, :self.expected_features]
            else:
                # Insufficient features, return input as is
                return X
        
        # Apply variance selector and feature selection
        try:
            X_var = self.variance_selector.transform(X)
            return X_var[:, self.indices]
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            # Return input as fallback
            return X

def perform_feature_selection(features, labels, n_features=None, method='variance'):
    """
    Perform feature selection to reduce dimensionality and remove noisy features
    
    Args:
        features: Feature array
        labels: Label array
        n_features: Number of features to select (default: 80% of original)
        method: Feature selection method ('variance', 'mutual_info', 'random_forest')
        
    Returns:
        Tuple of (selected_features, feature_selector)
    """
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    
    if n_features is None:
        n_features = int(features.shape[1] * 0.8)  # Default to keeping 80% of features
    
    # Store original feature count for reference
    original_feature_count = features.shape[1]
    logger.info(f"Original feature count: {original_feature_count}")
    
    # Calculate the variance of each feature
    variances = np.var(features, axis=0)
    min_variance = np.min(variances)
    max_variance = np.max(variances)
    median_variance = np.median(variances)
    
    logger.info(f"Feature variance stats - Min: {min_variance:.6f}, Median: {median_variance:.6f}, Max: {max_variance:.6f}")
    
    # Use a data-adaptive threshold that's a fraction of the median variance
    # This ensures at least some features will pass the threshold
    adaptive_threshold = min(0.001, median_variance * 0.1)
    logger.info(f"Using adaptive variance threshold: {adaptive_threshold:.6f}")
    
    # First, remove features with near-zero variance
    variance_selector = VarianceThreshold(threshold=adaptive_threshold)
    features_var_selected = variance_selector.fit_transform(features)
    
    logger.info(f"After variance threshold: {features_var_selected.shape[1]} features (removed {features.shape[1] - features_var_selected.shape[1]})")
    
    # If all features were removed, return the original features
    if features_var_selected.shape[1] == 0:
        logger.warning("All features removed by variance threshold! Using original features.")
        features_var_selected = features
        # Reset to a pass-through selector
        variance_selector = VarianceThreshold(threshold=0.0)
        variance_selector.fit(features)
    
    # Save variance selector n_features_in_ for future reference
    selector_expected_features = variance_selector.n_features_in_
    logger.info(f"Variance selector expects {selector_expected_features} input features")
    
    # Apply specified selection method
    if method == 'variance':
        # Just use variance threshold
        return features_var_selected, variance_selector
        
    elif method == 'mutual_info':
        # Use mutual information for feature selection
        k = min(n_features, features_var_selected.shape[1])
        info_selector = SelectKBest(mutual_info_classif, k=k)
        features_selected = info_selector.fit_transform(features_var_selected, labels)
        
        # Create a combined selector using an importable class
        combined_selector = FeatureSelector(variance_selector, info_selector.get_support())
        return features_selected, combined_selector
        
    elif method == 'random_forest':
        # Use Random Forest for feature importance-based selection
        # For CASIAv2, ensure class balance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(features_var_selected, labels)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create a simpler approach - just manually select top k features
        k = min(n_features, features_var_selected.shape[1])
        
        # Sort indices by importance
        indices = np.argsort(importances)[::-1][:k]
        
        # Create a manual selector that just picks these columns
        features_selected = features_var_selected[:, indices]
        
        # Create selector using the globally defined class
        combined_selector = RobustFeatureSelector(variance_selector, indices, selector_expected_features)
        
        # Print top features
        top_features = sorted(zip(range(len(indices)), importances[indices]), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 features by importance:")
        for i, (idx, importance) in enumerate(top_features):
            logger.info(f"  {i+1}. Feature {idx}: {importance:.4f}")
        
        return features_selected, combined_selector
    
    # Default fallback
    return features, lambda x: x