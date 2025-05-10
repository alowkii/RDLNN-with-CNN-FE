import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from utils import load_dataset, save_model
import os
from datetime import datetime

def train_model(X, y):
    """
    Train a Random Forest classifier on the extracted features.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        Trained model, test data for evaluation, and performance metrics
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define model with hyperparameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Initialize and train the model with grid search
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Save the scaler with the model to use during inference
    model_data = {
        'model': model,
        'scaler': scaler,
        'best_params': grid_search.best_params_
    }
    
    return model_data, (X_test_scaled, y_test, y_pred, y_prob), (report, conf_matrix, accuracy)

def save_report(report, conf_matrix, accuracy, output_dir="training_reports"):
    """
    Save the training report as text and visualization.
    
    Args:
        report: Classification report text
        conf_matrix: Confusion matrix
        accuracy: Model accuracy
        output_dir: Directory to save reports
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text report
    report_filename = os.path.join(output_dir, f"report_{timestamp}.txt")
    with open(report_filename, 'w') as f:
        f.write(f"Training Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    
    print(f"Text report saved to {report_filename}")
    
    # Create and save visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix
    im = ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax1)
    classes = ['Authentic', 'Forged']
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes)
    ax1.set_yticklabels(classes)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Add text annotations to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax1.text(j, i, conf_matrix[i, j],
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    # Plot performance metrics
    try:
        # Extract metrics from the report
        metrics = {
            'Accuracy': accuracy,
            'Precision (Forged)': float(report.split('\n')[3].split()[3]),
            'Recall (Forged)': float(report.split('\n')[3].split()[4]),
            'F1-Score (Forged)': float(report.split('\n')[3].split()[5])
        }
        
        ax2.bar(list(metrics.keys()), list(metrics.values()))
        ax2.set_title('Performance Metrics')
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel('Score')
        
        # Add value labels
        for i, v in enumerate(metrics.values()):
            ax2.text(i, v + 0.02, f'{v:.4f}', ha='center')
            
    except (IndexError, ValueError):
        # If parsing the report fails, just show accuracy
        ax2.bar(['Accuracy'], [accuracy])
        ax2.set_title('Model Accuracy')
        ax2.set_ylim(0, 1.0)
        ax2.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center')
    
    plt.tight_layout()
    
    # Save the figure
    viz_filename = os.path.join(output_dir, f"report_viz_{timestamp}.png")
    plt.savefig(viz_filename)
    plt.close()
    
    print(f"Visualization saved to {viz_filename}")
    
    # Create ROC curve if available
    try:
        if len(test_data) > 3:
            _, y_test, _, y_prob = test_data
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            
            # Save ROC curve
            roc_filename = os.path.join(output_dir, f"roc_curve_{timestamp}.png")
            plt.savefig(roc_filename)
            plt.close()
            
            print(f"ROC curve saved to {roc_filename}")
    except:
        # Skip ROC curve if there's an issue
        pass
    
    return report_filename, viz_filename

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Dyadic Wavelet Transform Image Forgery Detection Model')
    parser.add_argument('--authentic_dir', type=str, required=True,
                        help='Directory containing authentic images')
    parser.add_argument('--forged_dir', type=str, required=True,
                        help='Directory containing forged images')
    parser.add_argument('--model_path', type=str, default="dyadic_forgery_model.pkl",
                        help='Path to save the trained model')
    parser.add_argument('--report_dir', type=str, default="training_reports",
                        help='Directory to save training reports')
    parser.add_argument('--decomp_level', type=int, default=3,
                        help='Level of wavelet decomposition')
    
    args = parser.parse_args()
    
    print("Loading and extracting features from dataset...")
    X, y = load_dataset(args.authentic_dir, args.forged_dir, decomp_level=args.decomp_level)
    
    print(f"Dataset loaded: {len(X)} images, {np.sum(y)} forged, {len(X) - np.sum(y)} authentic")
    
    print("Training model...")
    model, test_data, performance = train_model(X, y)
    
    # Save the model
    save_model(model, args.model_path)
    
    # Save the report
    report, conf_matrix, accuracy = performance
    save_report(report, conf_matrix, accuracy, args.report_dir)

if __name__ == "__main__":
    main()