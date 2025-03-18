#!/usr/bin/env python3
"""
Extracts Polar Dyadic Wavelet Transform (PDyWT) features from images for forgery detection.
These features can be used to train the forgery detection model.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

from modules.pdywt_cnn import PDyWTCNNDetector
from modules.utils import logger, setup_logging


def resize_and_pad_image(img, target_size=(256, 256)):
    """
    Resize and pad image to ensure consistent dimensions
    
    Args:
        img: PIL Image
        target_size: Tuple of (height, width) for target size
        
    Returns:
        Resized and padded PIL Image
    """
    # Calculate aspect ratio
    width, height = img.size
    aspect_ratio = width / height
    
    # Determine new dimensions preserving aspect ratio
    if aspect_ratio > 1:  # Width > Height
        new_width = target_size[1]
        new_height = int(new_width / aspect_ratio)
    else:  # Height >= Width
        new_height = target_size[0]
        new_width = int(new_height * aspect_ratio)
    
    # Resize image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size (white background)
    padded_img = Image.new('RGB', target_size, (255, 255, 255))
    
    # Paste resized image centered on padded image
    paste_x = (target_size[1] - new_width) // 2
    paste_y = (target_size[0] - new_height) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))
    
    return padded_img


def extract_features_from_directory(directory, output_path, label=None, batch_size=32, num_workers=4, target_size=(256, 256)):
    """
    Extract PDyWT features from all images in a directory
    
    Args:
        directory: Directory containing images
        output_path: Path to save extracted features
        label: Label to assign to all images (0=authentic, 1=forged, None=unknown)
        batch_size: Batch size for processing
        num_workers: Number of parallel workers
        target_size: Target size for image resizing to ensure consistent feature dimensions
        
    Returns:
        Dictionary with image paths, features, and labels
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        logger.error(f"No image files found in {directory}")
        return None
    
    logger.info(f"Found {len(image_files)} images in {directory}")
    
    # Initialize result containers
    paths = []
    features_list = []
    labels_list = []
    
    # Initialize detector
    detector = PDyWTCNNDetector()
    
    # Track feature dimensions
    feature_dim = None
    
    # Define feature extraction function for parallel processing
    def extract_features_from_image(image_path):
        try:
            # Load and resize image to ensure consistent dimensions
            img = Image.open(image_path).convert("RGB")
            
            if img.size[0] > 2000 or img.size[1] > 2000:
                # Skip extremely large images
                logger.warning(f"Skipping large image: {image_path} ({img.size[0]}x{img.size[1]})")
                return None
                
            img = resize_and_pad_image(img, target_size)
            
            # Convert to numpy array and normalize
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Convert from RGB to YCbCr
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 128/255 - 0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 128/255 + 0.5 * r - 0.418688 * g - 0.081312 * b
            
            # Stack channels
            ycbcr = np.stack([y, cb, cr], axis=0)
            
            # Convert to tensor
            ycbcr_tensor = torch.tensor(ycbcr, dtype=torch.float32)
            
            # Extract wavelet features
            feature_tensor = detector.extract_wavelet_features(ycbcr_tensor)
            
            # Get a fixed-length feature vector by average pooling
            # This ensures consistent feature dimensions regardless of input size
            pooled_features = F.adaptive_avg_pool2d(feature_tensor.unsqueeze(0), (1, 1))
            flattened_features = pooled_features.view(-1).cpu().numpy()
            
            return flattened_features
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    # Process images in batches
    progress_bar = tqdm(total=len(image_files), desc="Extracting features", unit="img")
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        batch_paths = [os.path.join(directory, f) for f in batch_files]
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batch_features = list(executor.map(extract_features_from_image, batch_paths))
        
        # Filter out failed extractions
        valid_indices = [i for i, features in enumerate(batch_features) if features is not None]
        valid_paths = [batch_paths[i] for i in valid_indices]
        valid_features = [batch_features[i] for i in valid_indices]
        
        if valid_features:
            # Check and ensure consistent feature dimensions
            if feature_dim is None and valid_features:
                feature_dim = valid_features[0].shape[0]
                logger.info(f"Feature vector dimension: {feature_dim}")
            
            # Verify all features have the same dimension
            valid_features = [f for f in valid_features if f.shape[0] == feature_dim]
            
            # Update paths to only include valid features
            valid_paths = valid_paths[:len(valid_features)]
            
            paths.extend(valid_paths)
            features_list.extend(valid_features)
            
            if label is not None:
                labels_list.extend([label] * len(valid_features))
        
        progress_bar.update(len(batch_files))
    
    progress_bar.close()
    
    # Convert to numpy arrays
    if not features_list:
        logger.error("No valid features extracted")
        return None
    
    features_array = np.vstack(features_list)
    labels_array = np.array(labels_list) if labels_list else np.array([])
    
    # Save features to file
    np.savez(
        output_path,
        features=features_array,
        labels=labels_array,
        paths=paths
    )
    
    logger.info(f"Extracted {len(features_list)} feature vectors with shape {features_array.shape}")
    logger.info(f"Features saved to {output_path}")
    
    return {
        'features': features_array,
        'labels': labels_array,
        'paths': paths
    }


def combine_feature_files(authentic_path, forged_path, output_path):
    """
    Combine authentic and forged feature files into a single file
    
    Args:
        authentic_path: Path to authentic features file
        forged_path: Path to forged features file
        output_path: Path to save combined features
    """
    # Load authentic features
    logger.info(f"Loading authentic features from {authentic_path}")
    authentic_data = np.load(authentic_path, allow_pickle=True)
    authentic_features = authentic_data['features']
    authentic_paths = authentic_data['paths']
    authentic_labels = np.zeros(len(authentic_features), dtype=np.int32)
    
    # Load forged features
    logger.info(f"Loading forged features from {forged_path}")
    forged_data = np.load(forged_path, allow_pickle=True)
    forged_features = forged_data['features']
    forged_paths = forged_data['paths']
    forged_labels = np.ones(len(forged_features), dtype=np.int32)
    
    # Combine data
    combined_features = np.vstack([authentic_features, forged_features])
    combined_paths = np.concatenate([authentic_paths, forged_paths])
    combined_labels = np.concatenate([authentic_labels, forged_labels])
    
    # Save combined features
    np.savez(
        output_path,
        features=combined_features,
        labels=combined_labels,
        paths=combined_paths
    )
    
    logger.info(f"Combined features saved to {output_path}")
    logger.info(f"Total features: {len(combined_features)}")
    logger.info(f"Authentic: {len(authentic_features)}, Forged: {len(forged_features)}")


def main():
    parser = argparse.ArgumentParser(description="Extract PDyWT features for image forgery detection")
    
    parser.add_argument('--authentic_dir', type=str, help='Directory containing authentic images')
    parser.add_argument('--forged_dir', type=str, help='Directory containing forged images')
    parser.add_argument('--input_dir', type=str, help='Directory containing images to extract features from')
    
    parser.add_argument('--output_path', type=str, default='data/features/pdywt_features.npz',
                      help='Path to save the extracted features')
    
    parser.add_argument('--label', type=int, choices=[0, 1], default=None,
                      help='Label to assign to images (0=authentic, 1=forged)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of parallel workers')
    
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                      help='Target size for image resizing (height width)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(os.path.dirname(args.output_path) if args.output_path else '.')
    
    start_time = time.time()
    
    # If both authentic and forged directories are provided, extract and combine
    if args.authentic_dir and args.forged_dir:
        logger.info("Extracting features from authentic and forged images")
        
        # Extract authentic features
        authentic_output = os.path.splitext(args.output_path)[0] + '_authentic.npz'
        extract_features_from_directory(
            args.authentic_dir,
            authentic_output,
            label=0,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=tuple(args.target_size)
        )
        
        # Extract forged features
        forged_output = os.path.splitext(args.output_path)[0] + '_forged.npz'
        extract_features_from_directory(
            args.forged_dir,
            forged_output,
            label=1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=tuple(args.target_size)
        )
        
        # Combine features
        combine_feature_files(authentic_output, forged_output, args.output_path)
    
    # Otherwise, extract features from a single directory
    elif args.input_dir:
        logger.info(f"Extracting features from {args.input_dir}")
        extract_features_from_directory(
            args.input_dir,
            args.output_path,
            label=args.label,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=tuple(args.target_size)
        )
    
    else:
        parser.print_help()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()