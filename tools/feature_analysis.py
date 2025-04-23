#!/usr/bin/env python3
"""
Feature analysis and selection for forgery detection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse

# Import modules from existing codebase
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import logger, setup_logging

def load_and_verify_features(features_path):
    """
    Load and verify feature file
    
    Args:
        features_path: Path to the features file
        
    Returns:
        Tuple of (features, labels, paths)
    """
    logger.info(f"Loading features from {features_path}")
    
    try:
        data = np.load(features_path, allow_pickle=True)
        features = data['features']
        labels = data.get('labels', np.array([]))
        paths = data.get('paths', np.array([]))
        
        # Log basic info
        logger.info(f"Loaded {len(features)} feature vectors")
        if len(labels) > 0:
            logger.info(f"Class distribution: {np.bincount(labels.astype(int))}")
        
        return features, labels, paths
        
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return np.array([]), np.array([]), []

def analyze_feature_importance(features, labels, output_dir, n_top_features=20):
    """
    Analyze feature importance using Random Forest
    
    Args:
        features: Feature array
        labels: Label array
        output_dir: Directory to save results
        n_top_features: Number of top features to display
    
    Returns:
        Indices of most important features
    """
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features, labels)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    plt.bar(range(n_top_features), importances[indices[:n_top_features]])
    plt.xticks(range(n_top_features), indices[:n_top_features], rotation=90)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title(f'Top {n_top_features} Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
    
    # Print feature importance information
    logger.info(f"Top {n_top_features} features by importance:")
    for i in range(n_top_features):
        logger.info(f"  {i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    return indices[:n_top_features]

def visualize_feature_distribution(features, labels, output_dir):
    """
    Visualize feature distributions for authentic and forged images
    
    Args:
        features: Feature array
        labels: Label array
        output_dir: Directory to save visualizations
    """
    # Separate features by class
    authentic_features = features[labels == 0]
    forged_features = features[labels == 1]
    
    # Apply dimensionality reduction for visualization
    # PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(StandardScaler().fit_transform(features))
    
    # Plot PCA visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(features_pca[labels == 0, 0], features_pca[labels == 0, 1], 
               alpha=0.7, label='Authentic', c='blue')
    plt.scatter(features_pca[labels == 1, 0], features_pca[labels == 1, 1], 
               alpha=0.7, label='Forged', c='red')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
    
    # t-SNE for better visualization of high-dimensional data
    try:
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(StandardScaler().fit_transform(features))
        
        plt.figure(figsize=(10, 8))
        plt.scatter(features_tsne[labels == 0, 0], features_tsne[labels == 0, 1], 
                   alpha=0.7, label='Authentic', c='blue')
        plt.scatter(features_tsne[labels == 1, 0], features_tsne[labels == 1, 1], 
                   alpha=0.7, label='Forged', c='red')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    except Exception as e:
        logger.warning(f"Failed to generate t-SNE visualization: {e}")
    
    # Create histograms for top discriminating features
    feature_diff = np.abs(np.mean(authentic_features, axis=0) - np.mean(forged_features, axis=0))
    top_indices = np.argsort(-feature_diff)[:5]  # Top 5 discriminating features
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(top_indices):
        plt.subplot(2, 3, i+1)
        plt.hist(authentic_features[:, idx], bins=30, alpha=0.5, label='Authentic')
        plt.hist(forged_features[:, idx], bins=30, alpha=0.5, label='Forged')
        plt.title(f'Feature {idx}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_histograms.png'))

def main():
    parser = argparse.ArgumentParser(description="Feature analysis for image forgery detection")
    parser.add_argument('--features_path', type=str, required=True,
                      help='Path to features file')
    parser.add_argument('--output_dir', type=str, default='data/analysis',
                      help='Directory to save analysis results')
    parser.add_argument('--top_features', type=int, default=20,
                      help='Number of top features to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(args.output_dir)
    
    # Load features
    features, labels, paths = load_and_verify_features(args.features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return
    
    # Analyze feature importance
    top_features = analyze_feature_importance(features, labels, args.output_dir, args.top_features)
    
    # Visualize feature distributions
    visualize_feature_distribution(features, labels, args.output_dir)
    
    logger.info(f"Feature analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()