#!/usr/bin/env python3
"""
Generate Test Report for Image Forgery Detection Techniques

This script processes test results from the comparative testing and
generates a detailed report suitable for inclusion in the research paper.

Usage:
  python generate_test_report.py --results-dir /path/to/results --output-file report.md

Requirements:
  - Python 3.6+
  - pandas
  - matplotlib
  - numpy
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_test_results(results_dir):
    """Load test results from the output directory"""
    summary_path = os.path.join(results_dir, 'summary.json')
    
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        return None
        
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    # Load detailed results for each technique
    techniques = summary['techniques']
    detailed_results = {}
    
    for technique in techniques:
        tech_dir = os.path.join(results_dir, technique)
        metrics_path = os.path.join(tech_dir, f"{technique}_metrics.json")
        details_path = os.path.join(tech_dir, f"{technique}_details.csv")
        
        # Load metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                detailed_results[technique] = json.load(f)
        
        # Load details if available
        if os.path.exists(details_path):
            details_df = pd.read_csv(details_path)
            detailed_results[technique]['details_df'] = details_df
            
    # Load comparative results
    comp_dir = os.path.join(results_dir, 'comparative')
    comp_path = os.path.join(comp_dir, 'comparison.json')
    
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comparative = json.load(f)
        detailed_results['comparative'] = comparative
        
    # Load category results if available
    cat_dir = os.path.join(results_dir, 'categories')
    cat_path = os.path.join(cat_dir, 'category_results.json')
    
    if os.path.exists(cat_path):
        with open(cat_path, 'r') as f:
            categories = json.load(f)
        detailed_results['categories'] = categories
        
    # Load specialized test results if available
    spec_dir = os.path.join(results_dir, 'specialized')
    if os.path.exists(spec_dir):
        specialized = {}
        
        # Check for minimal forgery test
        min_forgery_path = os.path.join(spec_dir, 'min_forgery', 'min_forgery_results.json')
        if os.path.exists(min_forgery_path):
            with open(min_forgery_path, 'r') as f:
                specialized['min_forgery'] = json.load(f)
                
        # Check for complex background test
        complex_bg_path = os.path.join(spec_dir, 'complex_bg', 'complex_bg_results.json')
        if os.path.exists(complex_bg_path):
            with open(complex_bg_path, 'r') as f:
                specialized['complex_bg'] = json.load(f)
                
        # Check for compression test
        compression_path = os.path.join(spec_dir, 'compression', 'compression_results.json')
        if os.path.exists(compression_path):
            with open(compression_path, 'r') as f:
                specialized['compression'] = json.load(f)
                
        # Check for multiple forgery test
        multiple_path = os.path.join(spec_dir, 'multiple_forgery', 'multiple_forgery_results.json')
        if os.path.exists(multiple_path):
            with open(multiple_path, 'r') as f:
                specialized['multiple_forgery'] = json.load(f)
                
        detailed_results['specialized'] = specialized
    
    return {
        'summary': summary,
        'details': detailed_results
    }

def generate_markdown_report(results, output_file):
    """Generate markdown report from test results"""
    summary = results['summary']
    details = results['details']
    
    # Start building the report
    report = []
    report.append("# Image Forgery Detection Techniques - Test Results")
    report.append(f"\nTest conducted on: {summary['test_time']}")
    report.append(f"Dataset size: {summary['dataset_size']} images")
    report.append(f"Techniques evaluated: {', '.join([t.upper() for t in summary['techniques']])}")
    report.append("")
    
    # Summary table
    report.append("## Summary of Results")
    report.append("\nThe following table summarizes the key performance metrics for each technique:")
    report.append("")
    report.append("| Technique | Accuracy | Precision | Recall | F1 Score | Processing Time |")
    report.append("|-----------|----------|-----------|--------|----------|----------------|")
    
    for technique in summary['techniques']:
        tech_summary = summary['summary'][technique]
        # Get processing time from detailed results if available
        proc_time = f"{details[technique].get('avg_processing_time', 'N/A'):.3f}s" if 'avg_processing_time' in details[technique] else "N/A"
        
        report.append(f"| {technique.upper()} | {tech_summary['accuracy']:.4f} | {tech_summary['precision']:.4f} | {tech_summary['recall']:.4f} | {tech_summary['f1_score']:.4f} | {proc_time} |")
    
    report.append("")
    report.append(f"**Best performing technique**: {summary['best_technique'].upper()} with F1 score of {summary['summary'][summary['best_technique']]['f1_score']:.4f}")
    report.append("")
    
    # Detailed performance analysis
    report.append("## Detailed Performance Analysis")
    
    # Confusion matrices
    report.append("\n### Confusion Matrices")
    
    for technique in summary['techniques']:
        if 'confusion_matrix' in details[technique]:
            cm = details[technique]['confusion_matrix']
            # Format as a 2x2 table
            report.append(f"\n#### {technique.upper()}")
            report.append("")
            report.append("| | Predicted Authentic | Predicted Forged |")
            report.append("|----------------------|-------------------|-----------------|")
            report.append(f"| **Actual Authentic** | {cm[0][0]} | {cm[0][1]} |")
            report.append(f"| **Actual Forged**    | {cm[1][0]} | {cm[1][1]} |")
            report.append("")
    
    # ROC AUC values
    report.append("\n### ROC AUC Values")
    report.append("\nThe Area Under the Receiver Operating Characteristic Curve (ROC AUC) is a measure of how well a classifier can distinguish between classes:")
    report.append("")
    
    for technique in summary['techniques']:
        if 'roc_auc' in details[technique]:
            report.append(f"- {technique.upper()}: {details[technique]['roc_auc']:.4f}")
    
    report.append("\n![ROC Curves Comparison](comparative/roc_curves_comparison.png)")
    report.append("\n*Figure 1: ROC Curves for all techniques*")
    report.append("")
    
    # Performance by category
    if 'categories' in details:
        report.append("## Performance by Forgery Category")
        report.append("\nThe following tables show performance metrics across different forgery categories:")
        
        categories = details['categories']
        for category in categories:
            report.append(f"\n### {category.title()} Forgeries")
            report.append("")
            report.append("| Technique | Accuracy | Precision | Recall | F1 Score |")
            report.append("|-----------|----------|-----------|--------|----------|")
            
            for technique in summary['techniques']:
                if technique in categories[category]:
                    cat_results = categories[category][technique]
                    report.append(f"| {technique.upper()} | {cat_results['accuracy']:.4f} | {cat_results['precision']:.4f} | {cat_results['recall']:.4f} | {cat_results['f1_score']:.4f} |")
            
            report.append("")
    
    # Specialized test results
    if 'specialized' in details:
        report.append("## Specialized Test Results")
        
        # Minimal forgery detection
        if 'min_forgery' in details['specialized']:
            report.append("\n### Minimal Forgery Detection")
            report.append("\nThis test evaluates the minimum forgery size that each technique can reliably detect:")
            report.append("")
            report.append("| Technique | Minimum Detectable Size (% of image) |")
            report.append("|-----------|--------------------------------------|")
            
            min_forgery = details['specialized']['min_forgery']
            for technique in min_forgery:
                report.append(f"| {technique.upper()} | {min_forgery[technique]['min_size']*100:.1f}% |")
            
            report.append("")
        
        # Complex background test
        if 'complex_bg' in details['specialized']:
            report.append("\n### Complex Background Performance")
            report.append("\nThis test compares performance on images with simple vs. complex backgrounds:")
            report.append("")
            report.append("| Technique | Simple Background Accuracy | Complex Background Accuracy | Performance Drop |")
            report.append("|-----------|----------------------------|----------------------------|-----------------|")
            
            complex_bg = details['specialized']['complex_bg']
            for technique in complex_bg:
                simple_acc = complex_bg[technique]['simple_bg_accuracy']
                complex_acc = complex_bg[technique]['complex_bg_accuracy']
                drop = simple_acc - complex_acc
                report.append(f"| {technique.upper()} | {simple_acc:.4f} | {complex_acc:.4f} | {drop:.4f} |")
            
            report.append("")
        
        # Compression robustness
        if 'compression' in details['specialized']:
            report.append("\n### Robustness to JPEG Compression")
            report.append("\nThis test evaluates how well each technique performs as JPEG compression quality decreases:")
            report.append("")
            
            compression = details['specialized']['compression']
            levels = compression['compression_levels']
            
            # Create header row with quality levels
            header = "| Technique |"
            for level in levels:
                header += f" Quality {level} |"
            report.append(header)
            
            # Create separator row
            separator = "|-----------|" + "------------|" * len(levels)
            report.append(separator)
            
            # Add data for each technique
            for technique in summary['techniques']:
                if technique in compression:
                    row = f"| {technique.upper()} |"
                    for level in levels:
                        row += f" {compression[technique]['accuracy_by_level'][str(level)]:.4f} |"
                    report.append(row)
            
            report.append("")
        
        # Multiple forgery detection
        if 'multiple_forgery' in details['specialized']:
            report.append("\n### Multiple Forgery Detection")
            report.append("\nThis test evaluates how well each technique detects images with multiple forgeries:")
            report.append("")
            report.append("| Technique | Multiple Forgery Detection Rate |")
            report.append("|-----------|----------------------------------|")
            
            multiple = details['specialized']['multiple_forgery']
            for technique in multiple:
                report.append(f"| {technique.upper()} | {multiple[technique]['multiple_detection_rate']:.4f} |")
            
            report.append("")
    
    # Processing efficiency
    report.append("## Processing Efficiency")
    report.append("\nThe following table shows processing time statistics for each technique:")
    report.append("")
    report.append("| Technique | Average Time (s) | Standard Deviation (s) |")
    report.append("|-----------|------------------|-----------------------|")
    
    for technique in summary['techniques']:
        if 'avg_processing_time' in details[technique]:
            avg_time = details[technique]['avg_processing_time']
            std_time = details[technique]['std_processing_time']
            report.append(f"| {technique.upper()} | {avg_time:.4f} | {std_time:.4f} |")
    
    report.append("\n![Processing Time Comparison](comparative/processing_time_comparison.png)")
    report.append("\n*Figure 2: Average processing time for each technique*")
    report.append("")
    
    # Conclusion
    report.append("## Conclusion")
    report.append("\nBased on the comprehensive testing performed, the following conclusions can be drawn:")
    report.append("")
    
    # Generate conclusion based on results
    best_technique = summary['best_technique']
    best_accuracy = summary['summary'][best_technique]['accuracy']
    best_f1 = summary['summary'][best_technique]['f1_score']
    
    report.append(f"1. The {best_technique.upper()} technique demonstrates the best overall performance with an accuracy of {best_accuracy:.4f} and F1 score of {best_f1:.4f}.")
    
    # Add more conclusions based on available results
    if 'categories' in details:
        category_strengths = {}
        for category in details['categories']:
            best_f1_in_category = 0
            best_tech_in_category = ""
            for technique in details['categories'][category]:
                if details['categories'][category][technique]['f1_score'] > best_f1_in_category:
                    best_f1_in_category = details['categories'][category][technique]['f1_score']
                    best_tech_in_category = technique
            category_strengths[category] = (best_tech_in_category, best_f1_in_category)
        
        report.append(f"2. For specific forgery types, the following techniques perform best:")
        for category, (tech, score) in category_strengths.items():
            report.append(f"   - {category.title()}: {tech.upper()} (F1 score: {score:.4f})")
    
    # Add processing efficiency conclusion
    fastest_tech = min(summary['techniques'], 
                     key=lambda t: details[t].get('avg_processing_time', float('inf')) 
                     if 'avg_processing_time' in details[t] else float('inf'))
    
    if 'avg_processing_time' in details[fastest_tech]:
        report.append(f"3. In terms of computational efficiency, the {fastest_tech.upper()} technique is fastest with an average processing time of {details[fastest_tech]['avg_processing_time']:.4f} seconds per image.")
    
    # Add robustness conclusion if compression test exists
    if 'specialized' in details and 'compression' in details['specialized']:
        most_robust = max(summary['techniques'],
                        key=lambda t: details['specialized']['compression'][t]['accuracy_by_level']['25'] 
                        if t in details['specialized']['compression'] else 0)
        
        if most_robust in details['specialized']['compression']:
            low_quality_acc = details['specialized']['compression'][most_robust]['accuracy_by_level']['25']
            report.append(f"4. The {most_robust.upper()} technique shows the best robustness against JPEG compression, maintaining {low_quality_acc:.4f} accuracy even at 25% quality.")
    
    # Overall recommendation
    report.append(f"\nOverall recommendation: The {best_technique.upper()} technique provides the best balance of accuracy, robustness, and detection capability for general-purpose image forgery detection. However, if processing speed is a primary concern, {fastest_tech.upper()} may be more suitable despite its lower accuracy.")
    
    # Write the report to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
        
    print(f"Report generated and saved to {output_file}")
    return output_file

def generate_latex_report(results, output_file):
    """Generate LaTeX report from test results (for academic publications)"""
    summary = results['summary']
    details = results['details']
    
    # Start building the LaTeX report
    report = []
    report.append("\\section{Test Results}")
    report.append("\\label{sec:test_results}")
    
    report.append("\\subsection{Performance Metrics}")
    report.append("The three techniques (DWT, DyWT, and RDLNN) were evaluated on a dataset consisting of " + 
                 f"{summary['dataset_size']} images. Table~\\ref{{tab:performance_metrics}} summarizes the key performance metrics.")
    
    # Summary table in LaTeX
    report.append("\\begin{table}[ht]")
    report.append("\\centering")
    report.append("\\caption{Performance Metrics for Image Forgery Detection Techniques}")
    report.append("\\label{tab:performance_metrics}")
    report.append("\\begin{tabular}{lccccc}")
    report.append("\\hline")
    report.append("Technique & Accuracy & Precision & Recall & F1 Score & Avg. Time (s) \\\\")
    report.append("\\hline")
    
    for technique in summary['techniques']:
        tech_summary = summary['summary'][technique]
        proc_time = f"{details[technique].get('avg_processing_time', 'N/A'):.3f}" if 'avg_processing_time' in details[technique] else "N/A"
        
        report.append(f"{technique.upper()} & {tech_summary['accuracy']:.4f} & {tech_summary['precision']:.4f} & " +
                     f"{tech_summary['recall']:.4f} & {tech_summary['f1_score']:.4f} & {proc_time} \\\\")
    
    report.append("\\hline")
    report.append("\\end{tabular}")
    report.append("\\end{table}")
    
    # ROC curves
    report.append("\\subsection{ROC Analysis}")
    report.append("The Receiver Operating Characteristic (ROC) curves for all three techniques are shown in Figure~\\ref{fig:roc_curves}. " +
                 f"The RDLNN technique achieved the highest Area Under Curve (AUC) of {details['rdlnn'].get('roc_auc', 0):.4f}.")
    
    report.append("\\begin{figure}[ht]")
    report.append("\\centering")
    report.append("\\includegraphics[width=0.8\\textwidth]{roc_curves_comparison.png}")
    report.append("\\caption{ROC curves for DWT, DyWT, and RDLNN techniques}")
    report.append("\\label{fig:roc_curves}")
    report.append("\\end{figure}")
    
    # Category performance
    if 'categories' in details:
        report.append("\\subsection{Performance by Forgery Category}")
        report.append("The detection performance varies by forgery type. Table~\\ref{tab:forgery_categories} shows the " +
                     "F1 scores for each technique across different forgery categories.")
        
        report.append("\\begin{table}[ht]")
        report.append("\\centering")
        report.append("\\caption{F1 Scores by Forgery Category}")
        report.append("\\label{tab:forgery_categories}")
        
        # Get categories
        categories = list(details['categories'].keys())
        
        # Create table header
        report.append("\\begin{tabular}{l" + "c" * len(categories) + "}")
        report.append("\\hline")
        header = "Technique"
        for category in categories:
            header += f" & {category.title()}"
        header += " \\\\"
        report.append(header)
        report.append("\\hline")
        
        # Add data for each technique
        for technique in summary['techniques']:
            row = f"{technique.upper()}"
            for category in categories:
                if technique in details['categories'][category]:
                    f1 = details['categories'][category][technique]['f1_score']
                    row += f" & {f1:.4f}"
                else:
                    row += " & -"
            row += " \\\\"
            report.append(row)
        
        report.append("\\hline")
        report.append("\\end{tabular}")
        report.append("\\end{table}")
    
    # Processing time
    report.append("\\subsection{Computational Efficiency}")
    report.append("The processing time for each technique varies significantly. Figure~\\ref{fig:processing_time} illustrates " +
                 "the average processing time per image.")
    
    report.append("\\begin{figure}[ht]")
    report.append("\\centering")
    report.append("\\includegraphics[width=0.7\\textwidth]{processing_time_comparison.png}")
    report.append("\\caption{Average processing time comparison}")
    report.append("\\label{fig:processing_time}")
    report.append("\\end{figure}")
    
    # Specialized tests
    if 'specialized' in details and 'compression' in details['specialized']:
        report.append("\\subsection{Robustness to JPEG Compression}")
        report.append("The resilience of each technique to JPEG compression was evaluated. Table~\\ref{tab:compression_robustness} " +
                     "shows how accuracy declines as compression quality decreases.")
        
        report.append("\\begin{table}[ht]")
        report.append("\\centering")
        report.append("\\caption{Accuracy vs. JPEG Compression Quality}")
        report.append("\\label{tab:compression_robustness}")
        
        # Get compression levels
        levels = details['specialized']['compression']['compression_levels']
        
        # Create table header
        report.append("\\begin{tabular}{l" + "c" * len(levels) + "}")
        report.append("\\hline")
        header = "Technique"
        for level in levels:
            header += f" & Q{level}"
        header += " \\\\"
        report.append(header)
        report.append("\\hline")
        
        # Add data for each technique
        for technique in summary['techniques']:
            if technique in details['specialized']['compression']:
                row = f"{technique.upper()}"
                for level in levels:
                    acc = details['specialized']['compression'][technique]['accuracy_by_level'][str(level)]
                    row += f" & {acc:.4f}"
                row += " \\\\"
                report.append(row)
        
        report.append("\\hline")
        report.append("\\end{tabular}")
        report.append("\\end{table}")
    
    # Conclusion
    report.append("\\subsection{Conclusion}")
    best_technique = summary['best_technique']
    best_f1 = summary['summary'][best_technique]['f1_score']
    
    report.append("Based on our comprehensive evaluation, the following conclusions can be drawn:")
    report.append("\\begin{itemize}")
    report.append(f"\\item The {best_technique.upper()} technique demonstrates superior performance with an F1 score of {best_f1:.4f}.")
    
    # Add category-specific conclusions if available
    if 'categories' in details:
        report.append("\\item For specific forgery types:")
        report.append("\\begin{itemize}")
        for category in details['categories']:
            best_f1_in_category = 0
            best_tech_in_category = ""
            for technique in details['categories'][category]:
                if details['categories'][category][technique]['f1_score'] > best_f1_in_category:
                    best_f1_in_category = details['categories'][category][technique]['f1_score']
                    best_tech_in_category = technique
            report.append(f"\\item {category.title()}: {best_tech_in_category.upper()} performs best (F1 score: {best_f1_in_category:.4f})")
        report.append("\\end{itemize}")
    
    # Add processing time conclusion
    fastest_tech = min(summary['techniques'], 
                     key=lambda t: details[t].get('avg_processing_time', float('inf')) 
                     if 'avg_processing_time' in details[t] else float('inf'))
    
    if 'avg_processing_time' in details[fastest_tech]:
        report.append(f"\\item The {fastest_tech.upper()} technique offers the best computational efficiency, " +
                     f"processing images in {details[fastest_tech]['avg_processing_time']:.4f} seconds on average.")
    
    report.append("\\end{itemize}")
    
    # Write the LaTeX report to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
        
    print(f"LaTeX report generated and saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate Test Report for Image Forgery Detection')
    parser.add_argument('--results-dir', required=True, help='Directory containing test results')
    parser.add_argument('--output-file', default='test_report.md', help='Output file path')
    parser.add_argument('--format', choices=['markdown', 'latex'], default='markdown', help='Output format')
    
    args = parser.parse_args()
    
    # Load test results
    results = load_test_results(args.results_dir)
    
    if not results:
        print("Error loading test results. Exiting.")
        return 1
    
    # Generate report in the specified format
    if args.format == 'markdown':
        generate_markdown_report(results, args.output_file)
    else:
        generate_latex_report(results, args.output_file)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())