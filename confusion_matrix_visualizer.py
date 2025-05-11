#!/usr/bin/env python3
"""
Script to generate confusion matrices and advanced metrics for model comparison
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import glob

def load_results(results_dir):
    """Load and parse all test results from CSV files"""
    results = []
    
    # Find all CSV result files
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    for file_path in csv_files:
        # Extract model and dataset name from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        model_name = parts[0]
        
        # Find dataset part (between model name and "results.csv")
        dataset_name = '_'.join(parts[1:-1])
        
        # Load CSV data
        try:
            df = pd.read_csv(file_path)
            
            # Determine expected label from dataset name
            expected_label = 1 if 'forged' in dataset_name.lower() else 0
            
            # Collect results
            results.append({
                'model': model_name,
                'dataset': dataset_name,
                'expected_label': expected_label,
                'predictions': df['Prediction'].values,
                'filename': df['Filename'].values,
                'probabilities': df['Probability'].values if 'Probability' in df.columns else None
            })
            
            print(f"Loaded {len(df)} results from {filename}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def calculate_metrics(results):
    """Calculate performance metrics for each model/dataset combination"""
    metrics = []
    
    for result in results:
        # Ground truth is same label for all samples in dataset
        y_true = np.full_like(result['predictions'], result['expected_label'])
        y_pred = result['predictions']
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store metrics
        metrics.append({
            'model': result['model'],
            'dataset': result['dataset'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
    
    return metrics

def generate_confusion_matrices(metrics, output_dir):
    """Generate and save confusion matrix plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate one plot per model/dataset combination
    for metric in metrics:
        model = metric['model']
        dataset = metric['dataset']
        
        # Create confusion matrix
        cm = np.array([
            [metric['tn'], metric['fp']],
            [metric['fn'], metric['tp']]
        ])
        
        # Create plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'])
        
        plt.title(f"{model.upper()} - {dataset}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add metrics as text
        plt.figtext(0.5, 0.01, 
                   f"Accuracy: {metric['accuracy']:.4f}, Precision: {metric['precision']:.4f}, Recall: {metric['recall']:.4f}, F1: {metric['f1']:.4f}", 
                   ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"{model}_{dataset}_confusion.png"))
        plt.close()
    
    print(f"Generated {len(metrics)} confusion matrix plots in {output_dir}")

def generate_aggregate_metrics(metrics, output_dir):
    """Generate aggregate metrics by model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group metrics by model
    models = set(m['model'] for m in metrics)
    
    # Calculate average metrics per model
    avg_metrics = []
    for model in models:
        model_metrics = [m for m in metrics if m['model'] == model]
        
        avg_metrics.append({
            'model': model,
            'accuracy': np.mean([m['accuracy'] for m in model_metrics]),
            'precision': np.mean([m['precision'] for m in model_metrics]),
            'recall': np.mean([m['recall'] for m in model_metrics]),
            'specificity': np.mean([m['specificity'] for m in model_metrics]),
            'f1': np.mean([m['f1'] for m in model_metrics])
        })
    
    # Sort by accuracy
    avg_metrics.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Create a table
    data = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
    
    for m in avg_metrics:
        data.append([
            m['model'].upper(),
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['specificity']:.4f}",
            f"{m['f1']:.4f}"
        ])
    
    # Save as CSV
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(os.path.join(output_dir, 'aggregate_metrics.csv'), index=False)
    
    # Generate bar plots for key metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    plt.figure(figsize=(12, 8))
    
    for i, metric_key in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        
        # Extract data
        model_names = [m['model'].upper() for m in avg_metrics]
        metric_values = [m[metric_key] for m in avg_metrics]
        
        # Plot
        bars = plt.bar(model_names, metric_values)
        
        # Add values on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{value:.4f}',
                    ha='center', va='bottom')
        
        plt.title(metric_names[i])
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregate_metrics.png'))
    plt.close()
    
    print(f"Generated aggregate metrics in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate visual analytics for test results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing test results')
    parser.add_argument('--output_dir', type=str, default='metric_visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found")
        return
    
    print(f"Loaded {len(results)} result sets")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Generate confusion matrices
    generate_confusion_matrices(metrics, args.output_dir)
    
    # Generate aggregate metrics
    generate_aggregate_metrics(metrics, args.output_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()