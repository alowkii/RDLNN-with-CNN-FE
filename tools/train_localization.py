#!/usr/bin/env python3
"""
This script trains the localization model with a simplified architecture
that is more robust to different input sizes
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.feature_extractor import PDyWTCNNDetector
from modules.utils import logger, setup_logging
from modules.model import SimpleLocalizationModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define dataset class outside the function to avoid pickling issues
class FixedSizeLocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotations_dir, target_size=(128, 128)):
        self.data_dir = data_dir
        self.annotations_dir = annotations_dir
        self.target_size = target_size
        self.detector = PDyWTCNNDetector()
        
        # Get all image files from data_dir
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(glob.glob(os.path.join(data_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
        
        # Get all mask files from annotations_dir
        mask_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            mask_files.extend(glob.glob(os.path.join(annotations_dir, ext)))
            mask_files.extend(glob.glob(os.path.join(annotations_dir, ext.upper())))
        
        logger.info(f"Found {len(self.image_files)} images and {len(mask_files)} masks")
        
        # Create dataset with matched image-mask pairs
        self.matched_pairs = []
        for img_path in self.image_files:
            img_name = os.path.basename(img_path)
            img_base = os.path.splitext(img_name)[0]
            
            # Look for corresponding mask
            mask_path = None
            for mask_file in mask_files:
                mask_name = os.path.basename(mask_file)
                mask_base = os.path.splitext(mask_name)[0]
                
                # Check if names match or if mask name is contained in image name
                if mask_base == img_base or mask_base in img_base or img_base in mask_base:
                    mask_path = mask_file
                    break
            
            if mask_path:
                self.matched_pairs.append((img_path, mask_path))
        
        logger.info(f"Found {len(self.matched_pairs)} matched image-mask pairs")
        
    def __len__(self):
        return len(self.matched_pairs)
        
    def __getitem__(self, idx):
        img_path, mask_path = self.matched_pairs[idx]
        
        try:
            # Preprocess image
            ycbcr_tensor = self.detector.preprocess_image(img_path)
            if ycbcr_tensor is None:
                # Return a dummy tensor with the right dimensions if preprocessing fails
                return (torch.zeros(12, *self.target_size), 
                        torch.zeros(1, *self.target_size))
            
            # Extract wavelet features
            feature_tensor = self.detector.extract_wavelet_features(ycbcr_tensor)
            
            # Resize to target size to ensure consistent dimensions
            feature_tensor = F.interpolate(
                feature_tensor.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Process mask
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 0).astype(np.float32)  # Binarize
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
            # Resize mask to match feature dimensions
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=self.target_size,
                mode='nearest'
            ).squeeze(0)
            
            return feature_tensor, mask_tensor
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            # Return dummy tensors of the right size to avoid stopping the training
            return (torch.zeros(12, *self.target_size), 
                    torch.zeros(1, *self.target_size))

def train_localization_model(data_dir, annotations_dir, output_path, 
                           epochs=30, batch_size=4, learning_rate=0.0005):
    """
    Train the localization model with consistent tensor sizes
    """
    # Set up logging
    setup_logging(os.path.dirname(output_path))
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the new simpler localization model
    model = SimpleLocalizationModel(input_channels=12).to(device)
    
    # Create dataset and dataloader
    dataset = FixedSizeLocalizationDataset(data_dir, annotations_dir)
    
    # Check if there are any matched pairs
    if len(dataset.matched_pairs) == 0:
        logger.error("No matched image-mask pairs found. Check filenames.")
        return None
    
    # Use a small batch size to avoid memory issues
    # Set num_workers=0 to avoid multiprocessing issues
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True  # Drop incomplete batches
    )
    
    # Set model to training mode
    model.train()
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy for pixel-wise prediction
    
    # Training loop
    history = {'loss': [], 'dice_score': []}
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_dice = 0
        total_batches = 0
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_features, batch_masks in progress_bar:
            # Move to device
            batch_features = batch_features.to(device)
            batch_masks = batch_masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_features)
            
            # Calculate loss
            loss = criterion(predictions, batch_masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            
            # Calculate Dice coefficient (F1 score for segmentation)
            dice = calculate_dice(predictions, batch_masks)
            epoch_dice += dice.item()
            
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'dice': dice.item()
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / total_batches if total_batches > 0 else 0
        avg_dice = epoch_dice / total_batches if total_batches > 0 else 0
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Dice = {avg_dice:.4f}")
        
        # Store history
        history['loss'].append(avg_loss)
        history['dice_score'].append(avg_dice)
    
    # Save model
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(model.state_dict(), output_path)
        logger.info(f"Localization model saved to {output_path}")
    
    # Set model back to evaluation mode
    model.eval()
    
    return history

def calculate_dice(pred, target, smooth=1e-5):
    """Calculate Dice coefficient for binary segmentation"""
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum()
    return (2. * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train forgery localization model")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with tampered images')
    parser.add_argument('--annotations_dir', type=str, required=True, help='Directory with ground truth masks')
    parser.add_argument('--output_path', type=str, default='data/models/pdywt_localizer.pth', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    
    args = parser.parse_args()
    
    train_localization_model(
        data_dir=args.data_dir,
        annotations_dir=args.annotations_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )