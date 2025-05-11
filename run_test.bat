@echo off
REM Script to run the model testing on Windows

REM Set paths (update these to match your environment)
set TEST_DIR=data\test_datasets
set RDLNN_MODEL=data\models\rdlnn_model.pth
set DWT_MODEL=data\models\dwt_forgery_model.pkl
set DYWT_MODEL=data\models\dyadic_forgery_model.pkl
set OUTPUT_DIR=test_results
set THRESHOLD=0.675

REM Create output directory if it doesn't exist
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Run the test script
python test_models.py ^
  --test_dir %TEST_DIR% ^
  --rdlnn_model %RDLNN_MODEL% ^
  --dwt_model %DWT_MODEL% ^
  --dywt_model %DYWT_MODEL% ^
  --output_dir %OUTPUT_DIR% ^
  --visualize ^
  --threshold %THRESHOLD%

echo Testing complete. Results are saved in %OUTPUT_DIR%

REM Generate additional plot of accuracy by model type
python -c "^
import matplotlib.pyplot as plt^
import numpy as np^
import os^

# Read summary results^
summary_file = os.path.join('%OUTPUT_DIR%', 'summary.txt')^
with open(summary_file, 'r') as f:^
    lines = f.readlines()^

# Parse accuracy values^
models = []^
accuracies = []^

current_model = None^
for line in lines:^
    if 'results for' in line:^
        parts = line.split()^
        current_model = parts[0]^
    elif 'Accuracy:' in line:^
        accuracy = float(line.split(':')[1].strip())^
        models.append(current_model)^
        accuracies.append(accuracy)^

# Group by model type^
rdlnn_acc = [acc for model, acc in zip(models, accuracies) if 'RDLNN' in model]^
dwt_acc = [acc for model, acc in zip(models, accuracies) if 'DWT' in model]^
dywt_acc = [acc for model, acc in zip(models, accuracies) if 'DYWT' in model]^

# Calculate average accuracy per model^
model_names = ['RDLNN', 'DWT', 'DyWT']^
avg_accuracies = [^
    np.mean(rdlnn_acc) if rdlnn_acc else 0,^
    np.mean(dwt_acc) if dwt_acc else 0,^
    np.mean(dywt_acc) if dywt_acc else 0^
]^

# Plot average accuracy by model^
plt.figure(figsize=(10, 6))^
plt.bar(model_names, avg_accuracies, color=['blue', 'green', 'red'])^
plt.xlabel('Model')^
plt.ylabel('Average Accuracy')^
plt.title('Average Accuracy by Model Type')^
plt.ylim(0, 1.0)^
plt.grid(True, alpha=0.3)^

# Add value labels on bars^
for i, acc in enumerate(avg_accuracies):^
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')^

plt.tight_layout()^
plt.savefig(os.path.join('%OUTPUT_DIR%', 'avg_accuracy_by_model.png'))^
print('Generated average accuracy comparison plot')^
"

echo All visualizations complete!