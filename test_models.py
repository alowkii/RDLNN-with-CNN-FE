#!/usr/bin/env python3
"""
Comprehensive test script for image forgery detection models
Tests DWT, DyWT, and RDLNN with PDyWT models on multiple datasets

Usage:
    python test_models.py --model all
    python test_models.py --model dwt
    python test_models.py --model dywt
    python test_models.py --model rdlnn
"""

import os
import sys
import argparse
import time
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Import modules from the project
from dwt.main import detect_forgery as dwt_detect_forgery
from dwt.utils import load_model as dwt_load_model
from dywt.main import detect_forgery as dywt_detect_forgery
from dywt.utils import load_model as dywt_load_model
from modules.feature_extractor import PDyWTCNNDetector
from modules.rdlnn import RegressionDLNN
from modules.utils import setup_logging, logger, clean_cuda_memory

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def setup_directories():
    """Create necessary directories for results"""
    # Create results directory
    results_dir = os.path.join("data", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create model-specific directories
    model_dirs = ["dwt", "dywt", "rdlnn", "comparative"]
    for model_dir in model_dirs:
        os.makedirs(os.path.join(results_dir, model_dir), exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(logs_dir)
    logger.info("Starting comprehensive model testing")
    
    return results_dir

def parse_columbia(dataset_path):
    authentic = []
    forgery = []
    for img_file in glob.glob(os.path.join(dataset_path, "*.tif")):
        filename = os.path.basename(img_file).lower()
        if filename.startswith("au"):  # Authentic
            authentic.append(img_file)
        elif filename.startswith("tp"):  # Tampered
            forgery.append(img_file)
    return {"authentic": authentic, "forgery": forgery}


def find_test_datasets():
    """Find and organize test datasets"""
    # Look for dataset directories with format: data.test_datasets.NAME.TYPE
    datasets = {}
    data_dir = "data"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found")
        return datasets
    
    # Find test datasets
    test_datasets_dir = os.path.join(data_dir, "test_datasets")
    if not os.path.exists(test_datasets_dir):
        logger.error(f"Test datasets directory '{test_datasets_dir}' not found")
        return datasets
    
    # Get dataset names (subdirectories of test_datasets)
    dataset_names = [d for d in os.listdir(test_datasets_dir) 
                   if os.path.isdir(os.path.join(test_datasets_dir, d))]
    
    for dataset_name in dataset_names:
        dataset_path = os.path.join(test_datasets_dir, dataset_name)
        
        # Get types (authentic/forgery)
        type_dirs = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not type_dirs:
            logger.warning(f"No type directories found in {dataset_path}")
            continue
        
        datasets[dataset_name] = {}
        
        # Find authentic and forgery images
        for type_dir in type_dirs:
            if "authentic" in type_dir.lower():
                img_type = "authentic"
            elif "forgery" in type_dir.lower() or "forged" in type_dir.lower() or "tampered" in type_dir.lower():
                img_type = "forgery"
            else:
                logger.warning(f"Unknown image type '{type_dir}' in {dataset_name}")
                continue
            
            # Find images
            type_path = os.path.join(dataset_path, type_dir)
            image_files = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                image_files.update(glob.glob(os.path.join(type_path, ext)))
                image_files.update(glob.glob(os.path.join(type_path, ext.upper())))
            image_files = list(image_files)
            
            # Skip empty directories
            if not image_files:
                logger.warning(f"No images found in {type_path}")
                continue
            
            datasets[dataset_name][img_type] = image_files
            logger.info(f"Found {len(image_files)} {img_type} images in {dataset_name}")
    
    # Verify datasets
    valid_datasets = {}
    for dataset_name, dataset_types in datasets.items():
        if "authentic" in dataset_types and "forgery" in dataset_types:
            valid_datasets[dataset_name] = dataset_types
        else:
            logger.warning(f"Dataset {dataset_name} doesn't have both authentic and forgery images")
    
    return valid_datasets

def test_dwt_model(model_path, datasets, results_dir):
    """Test DWT model on all datasets"""
    logger.info(f"Testing DWT model: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"DWT model file not found: {model_path}")
        return None
    
    # Load the model
    try:
        model = dwt_load_model(model_path)
        logger.info("DWT model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading DWT model: {e}")
        return None
    
    # Results storage
    results = {
        "dataset": [],
        "image_path": [],
        "true_label": [],
        "predicted_label": [],
        "processing_time": []
    }
    
    # Test the model on each dataset
    for dataset_name, dataset_types in datasets.items():
        logger.info(f"Testing DWT model on {dataset_name} dataset")
        
        # Process authentic images
        for img_path in tqdm(dataset_types["authentic"], desc=f"DWT {dataset_name} (Authentic)"):
            start_time = time.time()
            prediction = dwt_detect_forgery(model, img_path)
            end_time = time.time()
            
            # Record result (convert prediction to integer)
            pred_value = int(prediction) if prediction is not None else -1
            
            results["dataset"].append(dataset_name)
            results["image_path"].append(img_path)
            results["true_label"].append(0)  # Authentic = 0
            results["predicted_label"].append(pred_value)
            results["processing_time"].append(end_time - start_time)
        
        # Process forgery images
        for img_path in tqdm(dataset_types["forgery"], desc=f"DWT {dataset_name} (Forgery)"):
            start_time = time.time()
            prediction = dwt_detect_forgery(model, img_path)
            end_time = time.time()
            
            # Record result (convert prediction to integer)
            pred_value = int(prediction) if prediction is not None else -1
            
            results["dataset"].append(dataset_name)
            results["image_path"].append(img_path)
            results["true_label"].append(1)  # Forgery = 1
            results["predicted_label"].append(pred_value)
            results["processing_time"].append(end_time - start_time)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = os.path.join(results_dir, "dwt", "results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"DWT results saved to {csv_path}")
    
    # Calculate and save metrics
    metrics = calculate_metrics(df)
    metrics_path = os.path.join(results_dir, "dwt", "metrics.csv")
    metrics.to_csv(metrics_path, index=True)
    logger.info(f"DWT metrics saved to {metrics_path}")
    
    # Generate visualization
    plot_path = os.path.join(results_dir, "dwt", "metrics_visualization.png")
    plot_metrics(metrics, "DWT Model Performance", plot_path)
    
    return df

def test_dywt_model(model_path, datasets, results_dir, threshold=0.5):
    """Test DyWT model on all datasets"""
    logger.info(f"Testing DyWT model: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"DyWT model file not found: {model_path}")
        return None
    
    # Load the model
    try:
        model = dywt_load_model(model_path)
        logger.info("DyWT model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading DyWT model: {e}")
        return None
    
    # Results storage
    results = {
        "dataset": [],
        "image_path": [],
        "true_label": [],
        "predicted_label": [],
        "confidence": [],
        "processing_time": []
    }
    
    # Test the model on each dataset
    for dataset_name, dataset_types in datasets.items():
        logger.info(f"Testing DyWT model on {dataset_name} dataset")
        
        # Process authentic images
        for img_path in tqdm(dataset_types["authentic"], desc=f"DyWT {dataset_name} (Authentic)"):
            start_time = time.time()
            result, probability = dywt_detect_forgery(model, img_path, threshold=threshold)
            end_time = time.time()
            
            # Record result (convert prediction to integer)
            pred_value = int(result) if result is not None else -1
            conf_value = float(probability) if probability is not None else 0.0
            
            results["dataset"].append(dataset_name)
            results["image_path"].append(img_path)
            results["true_label"].append(0)  # Authentic = 0
            results["predicted_label"].append(pred_value)
            results["confidence"].append(conf_value)
            results["processing_time"].append(end_time - start_time)
        
        # Process forgery images
        for img_path in tqdm(dataset_types["forgery"], desc=f"DyWT {dataset_name} (Forgery)"):
            start_time = time.time()
            result, probability = dywt_detect_forgery(model, img_path, threshold=threshold)
            end_time = time.time()
            
            # Record result (convert prediction to integer)
            pred_value = int(result) if result is not None else -1
            conf_value = float(probability) if probability is not None else 0.0
            
            results["dataset"].append(dataset_name)
            results["image_path"].append(img_path)
            results["true_label"].append(1)  # Forgery = 1
            results["predicted_label"].append(pred_value)
            results["confidence"].append(conf_value)
            results["processing_time"].append(end_time - start_time)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = os.path.join(results_dir, "dywt", "results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"DyWT results saved to {csv_path}")
    
    # Calculate and save metrics
    metrics = calculate_metrics(df)
    metrics_path = os.path.join(results_dir, "dywt", "metrics.csv")
    metrics.to_csv(metrics_path, index=True)
    logger.info(f"DyWT metrics saved to {metrics_path}")
    
    # Generate visualization
    plot_path = os.path.join(results_dir, "dywt", "metrics_visualization.png")
    plot_metrics(metrics, "DyWT Model Performance", plot_path)
    
    # Generate confidence histograms
    conf_plot_path = os.path.join(results_dir, "dywt", "confidence_histogram.png")
    plot_confidence_histogram(df, "DyWT Model Confidence Distribution", conf_plot_path)
    
    return df

def test_rdlnn_model(model_path, datasets, results_dir):
    """Test RDLNN model on all datasets"""
    logger.info(f"Testing RDLNN model: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"RDLNN model file not found: {model_path}")
        return None
    
    # Load the model
    try:
        # Initialize detector with PDyWT for feature extraction
        detector = PDyWTCNNDetector(model_path=model_path)
        logger.info("RDLNN model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading RDLNN model: {e}")
        return None
    
    # Results storage
    results = {
        "dataset": [],
        "image_path": [],
        "true_label": [],
        "predicted_label": [],
        "confidence": [],
        "processing_time": []
    }
    
    # Test the model on each dataset
    for dataset_name, dataset_types in datasets.items():
        logger.info(f"Testing RDLNN model on {dataset_name} dataset")
        
        # Process authentic images
        for img_path in tqdm(dataset_types["authentic"], desc=f"RDLNN {dataset_name} (Authentic)"):
            start_time = time.time()
            try:
                detection_result = detector.detect(img_path)
                prediction = detection_result.get('prediction', -1)
                confidence = detection_result.get('probability', 0.0)
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                prediction = -1
                confidence = 0.0
            end_time = time.time()
            
            results["dataset"].append(dataset_name)
            results["image_path"].append(img_path)
            results["true_label"].append(0)  # Authentic = 0
            results["predicted_label"].append(prediction)
            results["confidence"].append(confidence)
            results["processing_time"].append(end_time - start_time)
            
            # Periodically clean CUDA memory to avoid OOM errors
            if len(results["dataset"]) % 50 == 0:
                clean_cuda_memory()
        
        # Process forgery images
        for img_path in tqdm(dataset_types["forgery"], desc=f"RDLNN {dataset_name} (Forgery)"):
            start_time = time.time()
            try:
                detection_result = detector.detect(img_path)
                prediction = detection_result.get('prediction', -1)
                confidence = detection_result.get('probability', 0.0)
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                prediction = -1
                confidence = 0.0
            end_time = time.time()
            
            results["dataset"].append(dataset_name)
            results["image_path"].append(img_path)
            results["true_label"].append(1)  # Forgery = 1
            results["predicted_label"].append(prediction)
            results["confidence"].append(confidence)
            results["processing_time"].append(end_time - start_time)
            
            # Periodically clean CUDA memory to avoid OOM errors
            if len(results["dataset"]) % 50 == 0:
                clean_cuda_memory()
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = os.path.join(results_dir, "rdlnn", "results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"RDLNN results saved to {csv_path}")
    
    # Calculate and save metrics
    metrics = calculate_metrics(df)
    metrics_path = os.path.join(results_dir, "rdlnn", "metrics.csv")
    metrics.to_csv(metrics_path, index=True)
    logger.info(f"RDLNN metrics saved to {metrics_path}")
    
    # Generate visualization
    plot_path = os.path.join(results_dir, "rdlnn", "metrics_visualization.png")
    plot_metrics(metrics, "RDLNN Model Performance", plot_path)
    
    # Generate confidence histograms
    conf_plot_path = os.path.join(results_dir, "rdlnn", "confidence_histogram.png")
    plot_confidence_histogram(df, "RDLNN Model Confidence Distribution", conf_plot_path)
    
    return df

def calculate_metrics(results_df):
    """Calculate performance metrics from results DataFrame"""
    # Filter out failed predictions
    valid_df = results_df[results_df["predicted_label"] != -1]
    
    # Check if there are any valid predictions
    if len(valid_df) == 0:
        logger.warning("No valid predictions to calculate metrics")
        return pd.DataFrame()
    
    # Group by dataset
    datasets = valid_df["dataset"].unique()
    
    # Initialize metrics DataFrame
    metrics_data = []
    
    # Calculate overall metrics
    overall_metrics = calculate_dataset_metrics(valid_df, "Overall")
    metrics_data.append(overall_metrics)
    
    # Calculate metrics for each dataset
    for dataset in datasets:
        dataset_df = valid_df[valid_df["dataset"] == dataset]
        dataset_metrics = calculate_dataset_metrics(dataset_df, dataset)
        metrics_data.append(dataset_metrics)
    
    # Combine into DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("dataset", inplace=True)
    
    return metrics_df

def calculate_dataset_metrics(df, dataset_name):
    """Calculate metrics for a specific dataset"""
    # Binary classification metrics
    true_labels = df["true_label"].values
    pred_labels = df["predicted_label"].values
    
    # Calculate confusion matrix
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    
    # Calculate metrics
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = 0
    
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = 0
    
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    
    # Average processing time
    avg_time = df["processing_time"].mean()
    
    # Create metrics dictionary
    metrics = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "avg_processing_time": avg_time,
        "total_images": len(df)
    }
    
    return metrics

def plot_metrics(metrics_df, title, save_path):
    """Create visualization of metrics"""
    # Create a plot
    plt.figure(figsize=(12, 8))
    
    # Get metrics columns to plot
    metrics_to_plot = ["accuracy", "precision", "recall", "specificity", "f1_score"]
    
    # Get datasets
    datasets = metrics_df.index
    
    # Create bar positions
    bar_width = 0.15
    positions = np.arange(len(datasets))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot) / 2) * bar_width
        plt.bar(positions + offset, metrics_df[metric], width=bar_width, label=metric.capitalize())
    
    # Add labels and legend
    plt.xlabel("Dataset")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(positions, datasets, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot) / 2) * bar_width
        for j, value in enumerate(metrics_df[metric]):
            plt.text(positions[j] + offset, value + 0.02, f"{value:.2f}", 
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confidence_histogram(df, title, save_path):
    """Create histogram of confidence values"""
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Process data: separate by true label
    authentic_conf = df[df["true_label"] == 0]["confidence"]
    forged_conf = df[df["true_label"] == 1]["confidence"]
    
    # First subplot: Authentic images
    plt.subplot(1, 2, 1)
    plt.hist(authentic_conf, bins=20, alpha=0.7, color='green')
    plt.title("Authentic Images")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    # Second subplot: Forged images
    plt.subplot(1, 2, 2)
    plt.hist(forged_conf, bins=20, alpha=0.7, color='red')
    plt.title("Forged Images")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def compare_models(results_dict, results_dir):
    """Compare performance of all models and generate comparison plots"""
    # Create a unified metrics dataframe
    model_metrics = {}
    
    for model_name, results_df in results_dict.items():
        if results_df is not None:
            metrics = calculate_metrics(results_df)
            model_metrics[model_name] = metrics
    
    # Check if any valid metrics
    if not model_metrics:
        logger.error("No valid model metrics to compare")
        return
    
    # Generate comparative metrics for overall results
    comparison_metrics = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "Specificity": [],
        "F1 Score": [],
        "Processing Time (ms)": []
    }
    
    # Extract overall metrics for each model
    for model_name, metrics_df in model_metrics.items():
        if "Overall" in metrics_df.index:
            comparison_metrics["Model"].append(model_name.upper())
            comparison_metrics["Accuracy"].append(metrics_df.loc["Overall", "accuracy"])
            comparison_metrics["Precision"].append(metrics_df.loc["Overall", "precision"])
            comparison_metrics["Recall"].append(metrics_df.loc["Overall", "recall"])
            comparison_metrics["Specificity"].append(metrics_df.loc["Overall", "specificity"])
            comparison_metrics["F1 Score"].append(metrics_df.loc["Overall", "f1_score"])
            # Convert to milliseconds for better readability
            comparison_metrics["Processing Time (ms)"].append(metrics_df.loc["Overall", "avg_processing_time"] * 1000)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_metrics)
    
    # Save comparison metrics
    csv_path = os.path.join(results_dir, "comparative", "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Model comparison saved to {csv_path}")
    
    # Generate comparison plots
    # Metrics comparison
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"]
    plt.figure(figsize=(12, 6))
    
    # Create bar positions
    bar_width = 0.15
    positions = np.arange(len(comparison_df["Model"]))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot) / 2) * bar_width
        plt.bar(positions + offset, comparison_df[metric], width=bar_width, label=metric)
    
    # Add labels and legend
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Performance Metrics Comparison")
    plt.xticks(positions, comparison_df["Model"])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot) / 2) * bar_width
        for j, value in enumerate(comparison_df[metric]):
            plt.text(positions[j] + offset, value + 0.02, f"{value:.2f}", 
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparative", "metrics_comparison.png"), dpi=150)
    plt.close()
    
    # Processing time comparison
    plt.figure(figsize=(10, 5))
    plt.bar(comparison_df["Model"], comparison_df["Processing Time (ms)"], color='skyblue')
    plt.xlabel("Model")
    plt.ylabel("Average Processing Time (ms)")
    plt.title("Processing Time Comparison")
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, value in enumerate(comparison_df["Processing Time (ms)"]):
        plt.text(i, value + 1, f"{value:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparative", "processing_time_comparison.png"), dpi=150)
    plt.close()
    
    # Generate ROC curve comparison if confidence values are available
    models_with_confidence = {"dywt", "rdlnn"}
    
    # Create dataset-specific comparisons
    # Find common datasets across models
    all_datasets = set()
    for model_name, metrics_df in model_metrics.items():
        all_datasets.update(metrics_df.index)
    
    # Remove "Overall" from datasets
    if "Overall" in all_datasets:
        all_datasets.remove("Overall")
    
    # Generate comparison for each dataset
    for dataset in all_datasets:
        dataset_comparison = {
            "Model": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": []
        }
        
        for model_name, metrics_df in model_metrics.items():
            if dataset in metrics_df.index:
                dataset_comparison["Model"].append(model_name.upper())
                dataset_comparison["Accuracy"].append(metrics_df.loc[dataset, "accuracy"])
                dataset_comparison["Precision"].append(metrics_df.loc[dataset, "precision"])
                dataset_comparison["Recall"].append(metrics_df.loc[dataset, "recall"])
                dataset_comparison["F1 Score"].append(metrics_df.loc[dataset, "f1_score"])
        
        # Create DataFrame
        dataset_df = pd.DataFrame(dataset_comparison)
        
        # Save to CSV
        dataset_csv = os.path.join(results_dir, "comparative", f"{dataset}_comparison.csv")
        dataset_df.to_csv(dataset_csv, index=False)
        
        # Create bar chart
        plt.figure(figsize=(10, 5))
        
        # Create bar positions
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
        bar_width = 0.2
        positions = np.arange(len(dataset_df["Model"]))
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot) / 2) * bar_width
            plt.bar(positions + offset, dataset_df[metric], width=bar_width, label=metric)
        
        # Add labels and legend
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title(f"{dataset} Dataset - Model Comparison")
        plt.xticks(positions, dataset_df["Model"])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "comparative", f"{dataset}_comparison.png"), dpi=150)
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test image forgery detection models")
    parser.add_argument("--model", choices=["all", "dwt", "dywt", "rdlnn"], default="all",
                       help="Model to test")
    parser.add_argument("--dwt_model_path", default="dwt/model/dwt_model.pkl",
                       help="Path to DWT model")
    parser.add_argument("--dywt_model_path", default="dywt/model/dywt_model.pkl",
                       help="Path to DyWT model")
    parser.add_argument("--rdlnn_model_path", default="data/models/rdlnn_model.pth",
                       help="Path to RDLNN model")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold for models with probability output")
    parser.add_argument("--output_dir", default="data/test_results",
                       help="Directory to save test results")
    
    args = parser.parse_args()
    
    # Setup directories
    results_dir = setup_directories()
    
    # Find test datasets
    datasets = find_test_datasets()    
    
    if not datasets:
        logger.error("No valid test datasets found")
        return
    
    logger.info(f"Found {len(datasets)} test datasets")
    for dataset_name, dataset_types in datasets.items():
        logger.info(f"  {dataset_name}: {len(dataset_types.get('authentic', []))} authentic, {len(dataset_types.get('forgery', []))} forgery")
    
    # Initialize results dictionary
    results = {}
    
    # Test models based on user selection
    if args.model in ["all", "dwt"]:
        results["dwt"] = test_dwt_model(args.dwt_model_path, datasets, results_dir)
    
    if args.model in ["all", "dywt"]:
        results["dywt"] = test_dywt_model(args.dywt_model_path, datasets, results_dir, args.threshold)
    
    if args.model in ["all", "rdlnn"]:
        results["rdlnn"] = test_rdlnn_model(args.rdlnn_model_path, datasets, results_dir)
    
    # Compare models if multiple models were tested
    if len(results) > 1:
        compare_models(results, results_dir)
        logger.info("Model comparison completed")
    
    logger.info("Testing completed successfully")

if __name__ == "__main__":
    main()