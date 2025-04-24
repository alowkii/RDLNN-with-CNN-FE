#!/usr/bin/env python3
"""
Precision-tuned training for image forgery detection
Optimized to maximize precision while maintaining good F1 score
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from modules.rdlnn import RegressionDLNN
from modules.data_handling import load_and_verify_features
from modules.utils import setup_logging, logger, plot_diagnostic_curves

def precision_tuned_training(features_path, model_path, output_dir,
                           minority_ratio=0.8, pos_weight=10.0, neg_weight=1.0,
                           use_focal_loss=True, focal_gamma=2.0,
                           epochs=50, learning_rate=0.003, batch_size=32,
                           threshold=0.5, use_lr_scheduler=True):
    """
    Precision-focused training with enhanced balancing and fixed threshold
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        minority_ratio: Ratio of minority class samples after resampling
        pos_weight: Weight for positive class in loss function
        neg_weight: Weight for negative class to penalize false positives
        use_focal_loss: Whether to use focal loss to focus on difficult examples
        focal_gamma: Gamma parameter for focal loss
        epochs: Number of epochs to train
        learning_rate: Learning rate
        batch_size: Batch size
        threshold: Fixed classification threshold (0.60 for optimal precision/F1)
        
    Returns:
        Trained RegressionDLNN model
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Ensure threshold is not None
    if threshold is None:
        threshold = 0.60
        logger.info(f"No threshold provided, using default threshold: {threshold}")
    
    # Set up logging with reduced verbosity
    setup_logging(output_dir)
    
    logger.info(f"Starting precision-tuned training with threshold={threshold}")
    logger.info(f"Parameters: minority_ratio={minority_ratio}, pos_weight={pos_weight}, neg_weight={neg_weight}")
    logger.info(f"Using focal loss: {use_focal_loss} (gamma={focal_gamma})")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return None, None, None
    
    # Apply feature selection
    from modules.data_handling import perform_feature_selection
    logger.info("Performing feature selection...")
    selected_features, feature_selector = perform_feature_selection(
        features, labels, method='random_forest', n_features=int(features.shape[1] * 0.7)
    )
    
    logger.info(f"Selected {selected_features.shape[1]} features out of {features.shape[1]}")
    
    # Now we should use the selected features instead of original features
    features = selected_features
    
    # Create model
    input_dim = features.shape[1]  # Use the dimension of selected features!
    logger.info(f"Training model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim, architecture='deep') # Use deep architecture for better performance
    
    # Store the feature selector in the model
    model.feature_selector = feature_selector

    # Create optimizer
    optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler
    if use_lr_scheduler:
        # Create a combined warmup + cosine annealing scheduler
        def lr_lambda(epoch):
            # Warmup for first 5 epochs
            if epoch < 5:
                return epoch / 5
            # Cosine annealing after warmup
            return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (epochs - 5)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None
    
    # Calculate class distribution for better weighting
    class_counts = np.bincount(labels.astype(int))
    class_distribution = class_counts / np.sum(class_counts)
    logger.info(f"Class distribution: Negative={class_distribution[0]:.2f}, Positive={class_distribution[1]:.2f}")
    
    # Dynamically adjust weights based on class distribution if needed
    if class_distribution[1] < 0.3:  # If positive class is very underrepresented
        adjusted_pos_weight = max(pos_weight, 1.0 / class_distribution[1])
        logger.info(f"Adjusting positive weight to {adjusted_pos_weight} based on class distribution")
        pos_weight = adjusted_pos_weight
    
    # Split data into train/validation/test sets (60/20/20 split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.4, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Training: {len(X_train)} samples | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # Separate minority and majority classes in training set
    X_majority = X_train[y_train == 0]
    X_minority = X_train[y_train == 1]
    y_majority = y_train[y_train == 0]
    y_minority = y_train[y_train == 1]
    
    # Oversample minority class to the specified ratio
    minority_target_count = int(len(X_majority) * minority_ratio)
    
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, 
        y_minority,
        replace=True,
        n_samples=minority_target_count,
        random_state=42
    )

    # Add Gaussian noise to features for augmentation
    def augment_with_noise(features, std=0.02):
        noise = np.random.normal(0, std, features.shape)
        return features + noise

    # Augment majority and minority classes
    X_majority_augmented = np.vstack([X_majority, augment_with_noise(X_majority)])
    y_majority_augmented = np.concatenate([y_majority, y_majority])

    X_minority_augmented = np.vstack([X_minority_oversampled, 
                                    augment_with_noise(X_minority_oversampled)])
    y_minority_augmented = np.concatenate([y_minority_oversampled, 
                                        y_minority_oversampled])

    # Combine classes and use augmented data for training
    X_balanced = np.vstack([X_majority_augmented, X_minority_augmented])
    y_balanced = np.concatenate([y_majority_augmented, y_minority_augmented])
    
    # Shuffle
    indices = np.arange(len(X_balanced))
    np.random.shuffle(indices)
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    # Create model
    input_dim = features.shape[1]
    logger.info(f"Training model with input dimension: {input_dim}")
    
    # Create pos_weight tensor on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    
    # Define focal loss if requested
    if use_focal_loss:
        # Custom focal loss with class weights
        # Replace the FocalLoss class in training/precision.py
        class FocalLoss(torch.nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.pos_weight = pos_weight
                self.reduction = reduction
                self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                
            def forward(self, inputs, targets):
                # Calculate standard BCE loss
                bce_loss = self.bce_loss(inputs, targets)
                
                # Apply sigmoid to get probabilities
                inputs_sigmoid = torch.sigmoid(inputs)
                
                # Calculate focal term with different weighting for positive/negative examples
                p_t = torch.where(targets == 1, inputs_sigmoid, 1 - inputs_sigmoid)
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
                
                # Apply focal weighting
                focal_weight = alpha_t * (1 - p_t) ** self.gamma
                loss = focal_weight * bce_loss
                
                if self.reduction == 'mean':
                    return loss.mean()
                return loss.sum()
        
        # Create focal loss with balanced class weights
        logger.info(f"Using Focal Loss with gamma={focal_gamma}")
        model.loss_fn = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight_tensor)
    else:
        # Set custom loss function with specified pos_weight
        model.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Train model with more aggressive penalty for false positives
    logger.info("Starting training with enhanced balance...")
    
    # Add class_weights parameter to inform the model about the balance
    history = model.fit(
        X_balanced, y_balanced, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping=10,
        use_fp16=True,
        class_weights={0: neg_weight, 1: pos_weight},
        scheduler=scheduler  # Pass scheduler to fit method
    )
    
    # Evaluate with ensemble approach
    ensemble_preds, ensemble_confidences = ensemble_prediction(model, X_test, num_models=3, threshold=threshold)

    # Calculate metrics with ensemble predictions
    tp = np.sum((ensemble_preds == 1) & (y_test == 1))
    tn = np.sum((ensemble_preds == 0) & (y_test == 0))
    fp = np.sum((ensemble_preds == 1) & (y_test == 0))
    fn = np.sum((ensemble_preds == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    # Print results in a concise format
    logger.info(f"\n========== Performance with threshold={threshold:.2f} ==========")
    logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f} | Balanced Accuracy: {balanced_accuracy:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    
    # Print confusion matrix in a compact format
    logger.info("\nConfusion Matrix:")
    logger.info(f"TP: {tp} | FN: {fn}")
    logger.info(f"FP: {fp} | TN: {tn}")
    
    # Set threshold in the model
    model.threshold = threshold
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved with threshold {threshold} to {model_path}")
    
    # Create diagnostic plots if enabled
    try:
        plot_diagnostic_curves(y_test, ensemble_confidences, history, plots_dir)
        logger.info(f"Diagnostic plots saved to {plots_dir}")
    except Exception as e:
        logger.warning(f"Could not create diagnostic plots: {e}")
    
    # Evaluate with different thresholds for reference
    evaluate_with_thresholds(model, X_val, y_val, 
                           [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9], 
                           output_dir)
    
    return model, X_val, y_val

def evaluate_with_thresholds(model, X_val, y_val, thresholds, output_dir):
    """
    Evaluate model performance with different classification thresholds
    
    Args:
        model: Trained RegressionDLNN model
        X_val: Validation features
        y_val: Validation labels
        thresholds: List of threshold values to try
        output_dir: Directory to save threshold analysis results
    """
    # Get raw predictions
    _, confidences = model.predict(X_val)
    
    results = []
    logger.info("\nThreshold Analysis:")
    logger.info(f"{'Threshold':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10} | {'Accuracy':^10}")
    logger.info("-" * 60)
    
    best_f1 = 0
    best_threshold = 0.6
    
    for threshold in thresholds:
        # Apply threshold
        predictions = (confidences >= threshold).astype(int)
        
        # Compute metrics
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        tn = np.sum((predictions == 0) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        logger.info(f"{threshold:^10.2f} | {precision:^10.4f} | {recall:^10.4f} | {f1:^10.4f} | {accuracy:^10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"\nBest threshold based on F1: {best_threshold:.2f} (F1={best_f1:.4f})")
    
    # Create a threshold analysis plot
    try:
        thresholds_arr = np.array([result['threshold'] for result in results])
        precision_arr = np.array([result['precision'] for result in results])
        recall_arr = np.array([result['recall'] for result in results])
        f1_arr = np.array([result['f1'] for result in results])
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_arr, precision_arr, 'b-', label='Precision')
        plt.plot(thresholds_arr, recall_arr, 'g-', label='Recall')
        plt.plot(thresholds_arr, f1_arr, 'r-', label='F1 Score')
        plt.axvline(x=best_threshold, color='k', linestyle='--', alpha=0.3, label=f'Best F1 ({best_threshold:.2f})')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall and F1 Score vs. Threshold')
        plt.legend()
        
        # Save the plot
        threshold_plot_path = os.path.join(output_dir, 'threshold_analysis.png')
        plt.savefig(threshold_plot_path)
        logger.info(f"Threshold analysis plot saved to {threshold_plot_path}")
    except Exception as e:
        logger.warning(f"Could not create threshold analysis plot: {e}")
    
    return best_threshold, best_f1

def ensemble_prediction(model, X_test, num_models=3, threshold=0.5):
    """
    Make predictions using ensemble approach to improve stability
    
    Args:
        model: Trained RegressionDLNN model
        X_test: Test features
        num_models: Number of forward passes with dropout enabled
        threshold: Classification threshold
        
    Returns:
        Tuple of (predictions, confidences)
    """
    # Enable dropout during inference for MC Dropout
    model.model.train()
    
    all_confidences = []
    
    # Make multiple predictions with dropout enabled
    for _ in range(num_models):
        with torch.no_grad():
            # Get scaled inputs
            X_scaled = model.scaler.transform(X_test)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=model.device)
            
            # Get model output with dropout enabled
            logits = model.model(X_tensor)
            confidences = torch.sigmoid(logits).cpu().numpy().flatten()
            all_confidences.append(confidences)
    
    # Average predictions across ensemble
    ensemble_confidences = np.mean(all_confidences, axis=0)
    ensemble_preds = (ensemble_confidences >= threshold).astype(int)
    
    # Set model back to evaluation mode
    model.model.eval()
    
    return ensemble_preds, ensemble_confidences