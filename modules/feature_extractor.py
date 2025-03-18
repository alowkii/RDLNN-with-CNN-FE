#!/usr/bin/env python3
"""
Polar Dyadic Wavelet Transform (PDyWT) with CNN Feature Extraction - Fixed Version
For image forgery detection and localization
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Union, Optional

from modules.utils import logger, clean_cuda_memory


class PDyWaveletTransform:
    """
    Implements the Polar Dyadic Wavelet Transform for image decomposition.
    This transform combines traditional DWT with polar coordinates to better
    capture rotation invariant features which are useful for forgery detection.
    """
    
    def __init__(self, use_gpu: bool = True, decomposition_level: int = 2):
        """
        Initialize the PDyWT transformer
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            decomposition_level: Number of wavelet decomposition levels
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.decomposition_level = decomposition_level
        
        # Define Haar wavelet filters (simplest wavelet family)
        self.low_filter = torch.tensor([0.5, 0.5], dtype=torch.float32, device=self.device)
        self.high_filter = torch.tensor([0.5, -0.5], dtype=torch.float32, device=self.device)
        
        # Pre-compute 2D separable filters
        self._initialize_filters()
        
        logger.info(f"PDyWT initialized with decomposition level {decomposition_level} on {self.device}")

    def _initialize_filters(self):
        """Initialize 2D separable wavelet filters"""
        # Create 2D separable filters
        self.low_filter_x = self.low_filter.view(1, 1, 1, 2)
        self.low_filter_y = self.low_filter.view(1, 1, 2, 1)
        self.high_filter_x = self.high_filter.view(1, 1, 1, 2)
        self.high_filter_y = self.high_filter.view(1, 1, 2, 1)

    def transform(self, image_tensor: torch.Tensor) -> Dict[str, Tuple]:
        """
        Apply Polar Dyadic Wavelet Transform to an image
        
        Args:
            image_tensor: Image tensor of shape [C, H, W] in YCbCr color space
            
        Returns:
            Dictionary of wavelet coefficients for each channel
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
            
        if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
            raise ValueError(f"Expected input shape [3, H, W], got {image_tensor.shape}")

        # Convert to device if needed
        image_tensor = image_tensor.to(self.device)
        
        # Apply transform to each channel
        y_channel = image_tensor[0]
        cb_channel = image_tensor[1]
        cr_channel = image_tensor[2]
        
        # Apply PDyWT to each channel
        y_coeffs = self._apply_pdywt(y_channel)
        cb_coeffs = self._apply_pdywt(cb_channel) 
        cr_coeffs = self._apply_pdywt(cr_channel)
        
        return {
            'y': y_coeffs,
            'cb': cb_coeffs,
            'cr': cr_coeffs
        }
    
    def _apply_pdywt(self, channel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply PDyWT to a single channel
        
        Args:
            channel: Single channel tensor of shape [H, W]
            
        Returns:
            Tuple of (LL, LH, HL, HH) wavelet coefficients with polar coordinates applied
        """
        # Ensure even dimensions for wavelet transform
        h, w = channel.shape
        h_pad = 0 if h % 2 == 0 else 1
        w_pad = 0 if w % 2 == 0 else 1
        
        if h_pad != 0 or w_pad != 0:
            channel = F.pad(channel.unsqueeze(0).unsqueeze(0), 
                           (0, w_pad, 0, h_pad), mode='reflect').squeeze(0).squeeze(0)
        
        # Apply wavelet decomposition
        coeffs = []
        current = channel.unsqueeze(0).unsqueeze(0)
        
        for level in range(self.decomposition_level):
            # Apply DWT at current level
            LL, LH, HL, HH = self._dwt_level(current)
            
            # Apply polar transformation to wavelet coefficients
            LL_polar, LH_polar, HL_polar, HH_polar = self._transform_to_polar(LL, LH, HL, HH)
            
            # Save coefficients
            if level == 0:  # Save all coefficients from first level
                coeffs = [LL_polar, LH_polar, HL_polar, HH_polar]
            else:  # For deeper levels, replace LL with new coefficients
                coeffs[0] = LL_polar
                # Also store higher level detail coefficients
                coeffs.append(LH_polar)
                coeffs.append(HL_polar)
                coeffs.append(HH_polar)
            
            # Continue decomposition with LL coefficients
            current = LL
        
        return tuple(coeffs)
    
    def _dwt_level(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply one level of DWT using separable 2D convolution
        
        Args:
            x: Input tensor with shape [1, 1, H, W]
            
        Returns:
            Tuple of (LL, LH, HL, HH) wavelet coefficients
        """
        # Add proper padding
        pad_x = self.low_filter_x.shape[-1] // 2
        pad_y = self.low_filter_y.shape[-2] // 2
        x_padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
        
        # Apply filters
        LL = F.conv2d(F.conv2d(x_padded, self.low_filter_x, padding=0), 
                      self.low_filter_y, padding=0)
        LH = F.conv2d(F.conv2d(x_padded, self.low_filter_x, padding=0), 
                      self.high_filter_y, padding=0)
        HL = F.conv2d(F.conv2d(x_padded, self.high_filter_x, padding=0), 
                      self.low_filter_y, padding=0)
        HH = F.conv2d(F.conv2d(x_padded, self.high_filter_x, padding=0), 
                      self.high_filter_y, padding=0)
        
        # Downsample by strided convolution
        LL = F.avg_pool2d(LL, 2)
        LH = F.avg_pool2d(LH, 2)
        HL = F.avg_pool2d(HL, 2)
        HH = F.avg_pool2d(HH, 2)
        
        return LL, LH, HL, HH
    
    def _transform_to_polar(self, LL, LH, HL, HH):
        """
        Transform wavelet coefficients to polar coordinates for rotation invariance
        
        Args:
            LL, LH, HL, HH: Wavelet coefficients
            
        Returns:
            Polar-transformed coefficients
        """
        # Get coefficient shape
        _, _, h, w = LL.shape
        
        # Create coordinate grid
        y_coords = torch.arange(0, h, device=self.device).view(-1, 1).repeat(1, w)
        x_coords = torch.arange(0, w, device=self.device).view(1, -1).repeat(h, 1)
        
        # Calculate center point
        center_y, center_x = h // 2, w // 2
        
        # Calculate distance from center (radius)
        r = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Calculate angle
        theta = torch.atan2(y_coords - center_y, x_coords - center_x)
        
        # Normalize radius
        max_radius = torch.sqrt(torch.tensor((h/2)**2 + (w/2)**2, device=self.device))
        r_norm = r / max_radius
        
        # Apply polar weighting
        r_norm = r_norm.unsqueeze(0).unsqueeze(0)
        
        LL_polar = LL * r_norm
        LH_polar = LH * r_norm
        HL_polar = HL * r_norm
        HH_polar = HH * r_norm
        
        # Squeeze dimensions for consistency
        return (LL_polar.squeeze(0).squeeze(0),
                LH_polar.squeeze(0).squeeze(0),
                HL_polar.squeeze(0).squeeze(0),
                HH_polar.squeeze(0).squeeze(0))


# Modified MLP for flat feature vectors
class FlatFeatureDetector(nn.Module):
    """
    MLP for flat feature vector classification
    """
    
    def __init__(self, input_dim=12, hidden_dims=[128, 64, 32]):
        """
        Initialize the flat feature detector
        
        Args:
            input_dim: Number of input features
            hidden_dims: Dimensions of hidden layers
        """
        super(FlatFeatureDetector, self).__init__()
        
        # Build network layers
        layers = []
        
        # First layer with input dimension
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        
        # Final classification layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Logits and features
        """
        # Extract penultimate layer features
        features = None
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i == len(self.network) - 2:  # Features from penultimate layer
                features = x
        
        return x, features


class WaveletCNN(nn.Module):
    """
    CNN for feature extraction from wavelet coefficients
    """
    
    def __init__(self, input_channels=12, num_features=128):
        """
        Initialize the Wavelet CNN
        
        Args:
            input_channels: Number of input channels (default: 12 for 3 color channels × 4 subbands)
            num_features: Number of output features
        """
        super(WaveletCNN, self).__init__()
        
        # Feature extraction from wavelet coefficients
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global feature pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension reduction
        self.fc = nn.Linear(128, num_features)
        
        # Classification head
        self.classifier = nn.Linear(num_features, 1)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Classification logits and feature vectors
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Feature vector
        features = self.fc(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, features


class LocalizationCNN(nn.Module):
    """
    CNN for localization of forged regions in images.
    Uses a U-Net inspired architecture to generate pixel-wise predictions.
    """
    
    def __init__(self, input_channels=12):
        """
        Initialize the Localization CNN
        
        Args:
            input_channels: Number of input channels (default: 12 for 3 color channels × 4 subbands)
        """
        super(LocalizationCNN, self).__init__()
        
        # Encoder
        self.enc1 = self._encoder_block(input_channels, 32)
        self.enc2 = self._encoder_block(32, 64)
        self.enc3 = self._encoder_block(64, 128)
        self.enc4 = self._encoder_block(128, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.dec1 = self._decoder_block(512 + 256, 256)
        self.dec2 = self._decoder_block(256 + 128, 128)
        self.dec3 = self._decoder_block(128 + 64, 64)
        self.dec4 = self._decoder_block(64 + 32, 32)
        
        # Final output layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def _encoder_block(self, in_channels, out_channels):
        """Create an encoder block with double convolution and pooling"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        """Create a decoder block with upsampling and convolution"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """Forward pass with skip connections for U-Net architecture"""
        # Encoding
        enc1 = self.enc1(x)
        p1 = F.max_pool2d(enc1, 2)
        
        enc2 = self.enc2(p1)
        p2 = F.max_pool2d(enc2, 2)
        
        enc3 = self.enc3(p2)
        p3 = F.max_pool2d(enc3, 2)
        
        enc4 = self.enc4(p3)
        p4 = F.max_pool2d(enc4, 2)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        
        # Decoding with skip connections
        dec1 = self.dec1(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False), enc4], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=False), enc3], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False), enc2], 1))
        dec4 = self.dec4(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False), enc1], 1))
        
        # Final 1x1 convolution
        output = torch.sigmoid(self.final(dec4))
        
        return output


class PDyWTCNNDetector:
    """
    Main class for PDyWT-CNN based forgery detection and localization
    """
    
    def __init__(self, model_path=None, localization_model_path=None, use_gpu=True):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the pretrained WaveletCNN model
            localization_model_path: Path to the pretrained LocalizationCNN model
            use_gpu: Whether to use GPU acceleration if available
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.pdywt = PDyWaveletTransform(use_gpu=use_gpu)
        
        # Initialize detection model (auto-detect flat features model if needed)
        self.detection_model = None
        self.flat_feature_detector = None
        
        if model_path and os.path.exists(model_path):
            try:
                # First try loading as WaveletCNN
                self.detection_model = WaveletCNN().to(self.device)
                self.detection_model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded WaveletCNN detection model from {model_path}")
            except Exception as e:
                logger.info(f"Could not load as WaveletCNN, trying FlatFeatureDetector")
                # If that fails, try loading as FlatFeatureDetector
                try:
                    # Try to determine input dim from file
                    if model_path.endswith('.pth'):
                        checkpoint = torch.load(model_path, map_location=self.device)
                        # Check first layer weight shape to determine input dimension
                        input_dim = None
                        for key in checkpoint.keys():
                            if 'weight' in key and '0' in key:  # First layer weight
                                input_dim = checkpoint[key].shape[1]
                                break
                    
                    # Default to 12 if can't determine
                    if input_dim is None:
                        input_dim = 12
                    
                    self.flat_feature_detector = FlatFeatureDetector(input_dim=input_dim).to(self.device)
                    self.flat_feature_detector.load_state_dict(torch.load(model_path, map_location=self.device))
                    logger.info(f"Loaded FlatFeatureDetector model with input dim {input_dim} from {model_path}")
                except Exception as e2:
                    logger.error(f"Could not load detection model: {e2}")
        else:
            # Default to WaveletCNN if no model provided
            self.detection_model = WaveletCNN().to(self.device)
            logger.info(f"Initialized new WaveletCNN detection model")
        
        # Initialize localization model
        self.localization_model = LocalizationCNN().to(self.device)
        if localization_model_path and os.path.exists(localization_model_path):
            self.localization_model.load_state_dict(torch.load(localization_model_path, map_location=self.device))
            logger.info(f"Loaded localization model from {localization_model_path}")
        
        # Set to evaluation mode
        if self.detection_model:
            self.detection_model.eval()
        if self.flat_feature_detector:
            self.flat_feature_detector.eval()
        self.localization_model.eval()
        
        # Set threshold for classification
        self.threshold = 0.7
        logger.info(f"PDyWT-CNN detector initialized on {self.device}")
        
    def preprocess_image(self, image_path):
        """
        Preprocess an image for the detector
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor in YCbCr color space
        """
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            
            # Convert to numpy array
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Convert from RGB to YCbCr
            # Y = 0.299*R + 0.587*G + 0.114*B
            # Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
            # Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 128/255 - 0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 128/255 + 0.5 * r - 0.418688 * g - 0.081312 * b
            
            # Stack channels
            ycbcr = np.stack([y, cb, cr], axis=0)
            
            # Convert to tensor
            ycbcr_tensor = torch.tensor(ycbcr, dtype=torch.float32)
            
            return ycbcr_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def extract_wavelet_features(self, ycbcr_tensor):
        """
        Extract wavelet features from a YCbCr image tensor
        
        Args:
            ycbcr_tensor: YCbCr image tensor of shape [3, H, W]
            
        Returns:
            Wavelet feature tensor
        """
        # Apply PDyWT transform
        wavelet_coeffs = self.pdywt.transform(ycbcr_tensor)
        
        # Stack coefficients for feature extraction
        # We'll use the first level coefficients (LL, LH, HL, HH) from all channels
        y_coeffs = wavelet_coeffs['y']
        cb_coeffs = wavelet_coeffs['cb']
        cr_coeffs = wavelet_coeffs['cr']
        
        # Resize coefficients to match dimensions (LL might be smaller)
        target_size = y_coeffs[1].shape  # Use LH coefficient size as target
        
        def resize_tensor(tensor, size):
            # Resize coefficient tensor to target size
            return F.interpolate(tensor.unsqueeze(0).unsqueeze(0), 
                              size=size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        # Resize if needed and stack all coefficients
        feature_tensors = []
        for i in range(min(4, len(y_coeffs))):  # Use first 4 subbands from each channel
            # Get coefficient from each channel
            y_coeff = y_coeffs[i]
            cb_coeff = cb_coeffs[i]
            cr_coeff = cr_coeffs[i]
            
            # Resize if dimensions don't match
            if y_coeff.shape != target_size:
                y_coeff = resize_tensor(y_coeff, target_size)
            if cb_coeff.shape != target_size:
                cb_coeff = resize_tensor(cb_coeff, target_size)
            if cr_coeff.shape != target_size:
                cr_coeff = resize_tensor(cr_coeff, target_size)
            
            # Add to feature tensors
            feature_tensors.extend([y_coeff, cb_coeff, cr_coeff])
        
        # Stack into a single tensor [C, H, W]
        feature_tensor = torch.stack(feature_tensors[:12], dim=0)  # Take first 12 features
        
        return feature_tensor
    
    def detect(self, image_path):
        """
        Detect if an image is forged
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detection results
        """
        with torch.no_grad():
            # Preprocess image
            ycbcr_tensor = self.preprocess_image(image_path)
            if ycbcr_tensor is None:
                return {'error': 'Failed to preprocess image'}
            
            # Extract wavelet features
            feature_tensor = self.extract_wavelet_features(ycbcr_tensor)
            
            # Decide which model to use
            if self.detection_model:
                # Add batch dimension for CNN
                feature_tensor = feature_tensor.unsqueeze(0).to(self.device)
                
                # Run detection model
                logits, features = self.detection_model(feature_tensor)
                
            elif self.flat_feature_detector:
                # Flatten and pool for flat feature detector
                pooled_features = F.adaptive_avg_pool2d(feature_tensor.unsqueeze(0), (1, 1))
                flat_features = pooled_features.view(1, -1).to(self.device)
                
                # Run flat feature detector
                logits, features = self.flat_feature_detector(flat_features)
            
            else:
                return {'error': 'No detection model available'}
            
            # Calculate probability
            probability = torch.sigmoid(logits).item()
            
            # Make prediction
            prediction = 1 if probability >= self.threshold else 0
            
            # Return results
            return {
                'prediction': prediction,  # 1 = forged, 0 = authentic
                'probability': probability,
                'features': features.cpu().numpy()
            }
    
    def localize(self, image_path, save_path=None):
        """
        Localize potentially forged regions in an image
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the localization heatmap
            
        Returns:
            Dictionary with localization results and heatmap
        """
        with torch.no_grad():
            # Preprocess image
            ycbcr_tensor = self.preprocess_image(image_path)
            if ycbcr_tensor is None:
                return {'error': 'Failed to preprocess image'}
            
            # Extract wavelet features
            feature_tensor = self.extract_wavelet_features(ycbcr_tensor)
            
            # Add batch dimension
            feature_tensor = feature_tensor.unsqueeze(0).to(self.device)
            
            # Run localization model
            heatmap = self.localization_model(feature_tensor)
            
            # Convert heatmap to numpy array
            heatmap_np = heatmap.squeeze().cpu().numpy()
            
            # If the detection probability is high, also get region proposals
            region_proposals = []
            if self.detection_model or self.flat_feature_detector:
                # Get detection result
                detection_result = self.detect(image_path)
                probability = detection_result.get('probability', 0)
                
                if probability >= self.threshold:
                    # Find contours in the heatmap for region proposals
                    heatmap_img = (heatmap_np * 255).astype(np.uint8)
                    _, thresh = cv2.threshold(heatmap_img, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Convert contours to bounding boxes
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 20 and h > 20:  # Filter small regions
                            region_proposals.append({'x': x, 'y': y, 'width': w, 'height': h})
            
            # Save visualization if requested
            if save_path:
                self._save_localization_overlay(image_path, heatmap_np, save_path, region_proposals)
            
            # Return results
            return {
                'heatmap': heatmap_np,
                'region_proposals': region_proposals
            }
    
    def _save_localization_overlay(self, image_path, heatmap, save_path, regions=None):
        """Save visualization of localization results overlaid on the original image"""
        # Load original image
        original = np.array(Image.open(image_path).convert("RGB"))
        
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        # Normalize heatmap to [0, 1]
        heatmap_normalized = heatmap_resized / np.max(heatmap_resized) if np.max(heatmap_resized) > 0 else heatmap_resized
        
        # Create color heatmap
        heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original, 0.7, heatmap_colored, 0.3, 0)
        
        # Draw region proposals if available
        if regions:
            for region in regions:
                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the result
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved localization visualization to {save_path}")
    
    def train_detector(self, features_path, labels=None, output_path=None, 
                     epochs=20, batch_size=32, learning_rate=0.001):
        """
        Train the detection model on extracted features
        
        Args:
            features_path: Path to the extracted features
            labels: Array of labels (0 = authentic, 1 = forged)
            output_path: Path to save the trained model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            
        Returns:
            Training history
        """
        # Load features
        if isinstance(features_path, str):
            logger.info(f"Loading features from {features_path}")
            data = np.load(features_path, allow_pickle=True)
            features = data['features']
            if 'labels' in data and labels is None:
                labels = data['labels']
        else:
            features = features_path  # Assume features were passed directly
        
        if labels is None:
            raise ValueError("Labels must be provided either in the features file or as a parameter")
        
        # Check if features are already flat (1D) or need to be flattened
        if len(features.shape) > 2:  # More than 2D, need to flatten
            logger.info("Features need flattening/pooling")
            # Flatten features
            features_flat = features.reshape(features.shape[0], -1)
        else:
            logger.info("Features are already flat")
            features_flat = features
            
        logger.info(f"Feature shape: {features_flat.shape}")
        
        # Initialize flat feature detector if needed
        input_dim = features_flat.shape[1]
        if self.flat_feature_detector is None:
            logger.info(f"Initializing new FlatFeatureDetector with input dim {input_dim}")
            self.flat_feature_detector = FlatFeatureDetector(input_dim=input_dim).to(self.device)
        
        # Convert to tensors
        features_tensor = torch.tensor(features_flat, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set model to training mode
        self.flat_feature_detector.train()
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.flat_feature_detector.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        history = {'loss': [], 'accuracy': []}
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in dataloader:
                # Move to device
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                logits, _ = self.flat_feature_detector(batch_features)
                
                # Calculate loss
                loss = criterion(logits, batch_labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item() * batch_features.size(0)
                predictions = (torch.sigmoid(logits) >= 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / total
            accuracy = correct / total
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Store history
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
        
        # Save model
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(self.flat_feature_detector.state_dict(), output_path)
            logger.info(f"Model saved to {output_path}")
        
        # Set model back to evaluation mode
        self.flat_feature_detector.eval()
        
        return history
    
    def train_localizer(self, data_dir, annotations_dir, output_path, 
                       epochs=20, batch_size=8, learning_rate=0.001):
        """
        Train the localization model on annotated forgery data
        
        Args:
            data_dir: Directory containing image files
            annotations_dir: Directory containing annotation masks (binary masks of forged regions)
            output_path: Path to save the trained model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            
        Returns:
            Training history
        """
        import glob
        
        # Create feature extraction function
        def extract_features_from_image(img_path):
            try:
                # Preprocess image
                ycbcr_tensor = self.preprocess_image(img_path)
                if ycbcr_tensor is None:
                    logger.warning(f"Failed to preprocess image: {img_path}")
                    return None, None
                
                # Extract wavelet features
                feature_tensor = self.extract_wavelet_features(ycbcr_tensor)
                return feature_tensor, img_path
            except Exception as e:
                logger.error(f"Error extracting features from {img_path}: {e}")
                return None, None
        
        # Get all image files from data_dir
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(data_dir, ext)))
            image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
        
        # Get all mask files from annotations_dir
        mask_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            mask_files.extend(glob.glob(os.path.join(annotations_dir, ext)))
            mask_files.extend(glob.glob(os.path.join(annotations_dir, ext.upper())))
        
        logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
        
        # Create dataset with matched image-mask pairs
        matched_pairs = []
        for img_path in image_files:
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
                matched_pairs.append((img_path, mask_path))
        
        logger.info(f"Found {len(matched_pairs)} matched image-mask pairs")
        
        # No matches found
        if len(matched_pairs) == 0:
            logger.error("No matched image-mask pairs found. Check filenames.")
            return None
            
        # Create dataset
        class LocalizationDataset(torch.utils.data.Dataset):
            def __init__(self, matched_pairs, extract_func):
                self.matched_pairs = matched_pairs
                self.extract_func = extract_func
                
            def __len__(self):
                return len(self.matched_pairs)
                
            def __getitem__(self, idx):
                img_path, mask_path = self.matched_pairs[idx]
                
                # Extract features
                features, _ = self.extract_func(img_path)
                
                # Process mask
                mask = np.array(Image.open(mask_path).convert('L'))
                mask = (mask > 0).astype(np.float32)  # Binarize
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                
                # Resize mask to match feature dimensions
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), 
                                         size=features.shape[1:], 
                                         mode='nearest').squeeze(0)
                
                return features, mask_tensor
        
        # Create dataset and dataloader
        dataset = LocalizationDataset(matched_pairs, extract_features_from_image)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set model to training mode
        self.localization_model.train()
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.localization_model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()  # Binary Cross Entropy for pixel-wise prediction
        
        # Training loop
        history = {'loss': [], 'dice_score': []}
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_dice = 0
            total_batches = 0
            
            for batch_features, batch_masks in dataloader:
                # Move to device
                batch_features = batch_features.to(self.device)
                batch_masks = batch_masks.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.localization_model(batch_features)
                
                # Calculate loss
                loss = criterion(predictions, batch_masks)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                
                # Calculate Dice coefficient (F1 score for segmentation)
                dice = self._calculate_dice(predictions, batch_masks)
                epoch_dice += dice
                
                total_batches += 1
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / total_batches
            avg_dice = epoch_dice / total_batches
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Dice = {avg_dice:.4f}")
            
            # Store history
            history['loss'].append(avg_loss)
            history['dice_score'].append(avg_dice)
        
        # Save model
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(self.localization_model.state_dict(), output_path)
            logger.info(f"Localization model saved to {output_path}")
        
        # Set model back to evaluation mode
        self.localization_model.eval()
        
        return history
    
    def _calculate_dice(self, pred, target, smooth=1e-5):
        """Calculate Dice coefficient for binary segmentation"""
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * target).sum()
        return (2. * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)


def demo_forgery_detection(image_path, model_path=None, localization_model_path=None, output_dir='results'):
    """
    Demonstration of forgery detection and localization
    
    Args:
        image_path: Path to the image file to analyze
        model_path: Path to pretrained detection model
        localization_model_path: Path to pretrained localization model
        output_dir: Directory to save visualization results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = PDyWTCNNDetector(
        model_path=model_path,
        localization_model_path=localization_model_path
    )
    
    # Perform forgery detection
    detection_result = detector.detect(image_path)
    
    # Print detection results
    if 'error' in detection_result:
        print(f"Error: {detection_result['error']}")
        return
    
    prediction = "FORGED" if detection_result['prediction'] == 1 else "AUTHENTIC"
    confidence = detection_result['probability']
    print(f"Detection Result: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    # Perform localization if image is predicted as forged
    if detection_result['prediction'] == 1:
        print("Localizing potentially forged regions...")
        output_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_forgery_map.png')
        localization_result = detector.localize(image_path, save_path=output_path)
        
        if 'error' in localization_result:
            print(f"Localization error: {localization_result['error']}")
        else:
            print(f"Localization result saved to {output_path}")
            
            if localization_result['region_proposals']:
                print(f"Found {len(localization_result['region_proposals'])} suspicious regions.")
                for i, region in enumerate(localization_result['region_proposals']):
                    print(f"Region {i+1}: x={region['x']}, y={region['y']}, width={region['width']}, height={region['height']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDyWT-CNN Image Forgery Detection")
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default=None, help='Path to the detection model')
    parser.add_argument('--localization_model', type=str, default=None, help='Path to the localization model')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    demo_forgery_detection(
        args.image,
        model_path=args.model,
        localization_model_path=args.localization_model,
        output_dir=args.output_dir
    )