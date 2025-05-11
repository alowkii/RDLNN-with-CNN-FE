@echo off
REM Complete model testing workflow for Windows - Fixed version

REM Configuration variables
set TEST_DIR=data\test_datasets
set OUTPUT_DIR=test_results
set RDLNN_MODEL=data\models\rdlnn_model.pth
set DWT_MODEL=dwt\model\dwt_forgery_model.pkl
set DYWT_MODEL=dywt\model\dyadic_forgery_model.pkl

REM Create output directories
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
if not exist %OUTPUT_DIR%\metrics mkdir %OUTPUT_DIR%\metrics
if not exist %OUTPUT_DIR%\images mkdir %OUTPUT_DIR%\images

echo =============================
echo Image Forgery Model Testing
echo =============================

REM Step 1: Run the fixed test script instead of the original
echo Step 1: Running main test script...
python test_models.py ^
  --test_dir %TEST_DIR% ^
  --rdlnn_model %RDLNN_MODEL% ^
  --dwt_model %DWT_MODEL% ^
  --dywt_model %DYWT_MODEL% ^
  --output_dir %OUTPUT_DIR% ^
  --visualize ^
  --threshold 0.675

REM Skip confusion matrix generator until we have results
echo Results generated! Check the %OUTPUT_DIR% directory.

REM Generate the basic comparison report using Python
echo Generating basic report...

python -c "
import os
import matplotlib.pyplot as plt
import numpy as np

# Read summary results
output_dir = '%OUTPUT_DIR%'
summary_file = os.path.join(output_dir, 'summary.txt')

if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    # Parse accuracy values
    models = []
    accuracies = []
    
    current_model = None
    for line in lines:
        if 'results for' in line:
            parts = line.split()
            if len(parts) > 0:
                current_model = parts[0]
        elif 'Accuracy:' in line:
            try:
                accuracy = float(line.split(':')[1].strip())
                if current_model:
                    models.append(current_model)
                    accuracies.append(accuracy)
            except:
                pass
    
    # Group by model type
    rdlnn_acc = [acc for model, acc in zip(models, accuracies) if 'RDLNN' in model]
    dwt_acc = [acc for model, acc in zip(models, accuracies) if 'DWT' in model]
    dywt_acc = [acc for model, acc in zip(models, accuracies) if 'DYWT' in model]
    
    # Create simple report
    report_file = os.path.join(output_dir, 'model_comparison_report.txt')
    with open(report_file, 'w') as f:
        f.write('Model Testing Results Summary\n')
        f.write('============================\n\n')
        
        f.write('RDLNN Model Results:\n')
        if rdlnn_acc:
            avg_acc = np.mean(rdlnn_acc)
            f.write(f'  Average Accuracy: {avg_acc:.4f}\n')
            f.write(f'  Number of Tests: {len(rdlnn_acc)}\n\n')
        else:
            f.write('  No results available\n\n')
        
        f.write('DWT Model Results:\n')
        if dwt_acc:
            avg_acc = np.mean(dwt_acc)
            f.write(f'  Average Accuracy: {avg_acc:.4f}\n')
            f.write(f'  Number of Tests: {len(dwt_acc)}\n\n')
        else:
            f.write('  No results available\n\n')
        
        f.write('DyWT Model Results:\n')
        if dywt_acc:
            avg_acc = np.mean(dywt_acc)
            f.write(f'  Average Accuracy: {avg_acc:.4f}\n')
            f.write(f'  Number of Tests: {len(dywt_acc)}\n\n')
        else:
            f.write('  No results available\n\n')
        
        f.write('See individual CSV files for detailed results by dataset.')
    
    # If we have enough data, create a summary plot
    if rdlnn_acc or dwt_acc or dywt_acc:
        plt.figure(figsize=(10, 6))
        
        # Calculate average accuracy per model
        model_names = []
        avg_accuracies = []
        
        if rdlnn_acc:
            model_names.append('RDLNN')
            avg_accuracies.append(np.mean(rdlnn_acc))
        
        if dwt_acc:
            model_names.append('DWT')
            avg_accuracies.append(np.mean(dwt_acc))
        
        if dywt_acc:
            model_names.append('DyWT')
            avg_accuracies.append(np.mean(dywt_acc))
        
        # Plot average accuracy by model
        bars = plt.bar(model_names, avg_accuracies, color=['blue', 'green', 'red'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, avg_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.4f}', ha='center')
        
        plt.xlabel('Model')
        plt.ylabel('Average Accuracy')
        plt.title('Average Accuracy by Model Type')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'avg_accuracy_by_model.png'))
        print('Generated average accuracy comparison plot')
else:
    print('No summary file found to generate report')
"

echo Testing workflow complete!