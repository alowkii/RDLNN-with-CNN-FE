# Image Forgery Detection System

This repository contains a comprehensive framework for detecting and localizing image forgeries using deep learning and wavelet-based feature extraction techniques. The system leverages Polar Dyadic Wavelet Transform (PDyWT) with Convolutional Neural Networks (CNNs) and Regression Deep Learning Neural Networks (RDLNN) to identify manipulated images.

## Features

- **Image Forgery Detection**: Classify images as authentic or forged with high precision
- **Forgery Localization**: Identify and highlight specific regions of an image that have been tampered with
- **Multiple Wavelet Transforms**: Support for traditional Discrete Wavelet Transform (DWT), Dyadic Wavelet Transform (DyWT), and Polar Dyadic Wavelet Transform (PDyWT)
- **RDLNN Model Training**: Train specialized regression models optimized for imbalanced datasets
- **REST API Server**: Deploy the detection system as a web service with comprehensive endpoints
- **Batch Processing**: Efficiently process large image collections

## Installation

### Prerequisites

- Python 3.6+
- PyTorch 1.7+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/image-forgery-detection.git
   cd image-forgery-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p data/models data/features data/results
   ```

## Usage

### Feature Extraction

#### PDyWT Features (Main Method)

Extract PDyWT features from authentic and forged images:

```bash
python tools/extract_pdywt_features.py \
  --authentic_dir path/to/authentic/images \
  --forged_dir path/to/forged/images \
  --output_path data/features/pdywt_features.npz
```

Or extract features for a single directory:

```bash
python tools/extract_pdywt_features.py \
  --input_dir path/to/images \
  --output_path data/features/test_features.npz
```

#### DWT Features

Use the Discrete Wavelet Transform implementation:

```bash
python dwt/main.py train \
  --authentic_dir path/to/authentic/images \
  --forged_dir path/to/forged/images \
  --model_path dwt_forgery_model.pkl
```

#### DyWT Features

Use the Dyadic Wavelet Transform implementation:

```bash
python dywt/main.py train \
  --authentic_dir path/to/authentic/images \
  --forged_dir path/to/forged/images \
  --model_path dyadic_forgery_model.pkl \
  --decomp_level 3
```

### Training

Train a forgery detection model using different training approaches:

#### Precision-Tuned Training (Recommended)

This method optimizes for precision while maintaining a good F1 score:

```bash
python -m training.precision \
  --features_path data/features/pdywt_features.npz \
  --model_path data/models/rdlnn_model.pth \
  --output_dir data/results \
  --threshold 0.8 \
  --epochs 25
```

#### Alternative Training Methods

```bash
python -m training.balanced \
  --method combined \
  --features_path data/features/pdywt_features.npz \
  --model_path data/models/balanced_model.pth \
  --output_dir data/results
```

### Localization Model Training

Train the forgery localization model:

```bash
python tools/train_localization.py \
  --data_dir path/to/forged/images \
  --annotations_dir path/to/forgery/masks \
  --output_path data/models/pdywt_localizer.pth \
  --epochs 30
```

### Testing

#### Main Testing Method

Test the model on individual images or directories:

```bash
python main.py \
  --mode single \
  --image_path path/to/test/image.jpg \
  --model_path data/models/rdlnn_model.pth \
  --localization_model_path data/models/pdywt_localizer.pth \
  --output_dir data/results \
  --localize
```

For batch testing on a directory:

```bash
python main.py \
  --mode test \
  --input_dir path/to/test/images \
  --model_path data/models/rdlnn_model.pth \
  --output_dir data/results
```

#### DWT Testing

Test images using the Discrete Wavelet Transform model:

```bash
python dwt/main.py test \
  --model_path dwt_forgery_model.pkl \
  --test_image path/to/image.jpg
```

#### DyWT Testing

Test images using the Dyadic Wavelet Transform model:

```bash
python dywt/main.py test \
  --model_path dyadic_forgery_model.pkl \
  --test_image path/to/image.jpg \
  --threshold 0.5
```

Or batch test a directory:

```bash
python dywt/main.py batch_test \
  --model_path dyadic_forgery_model.pkl \
  --test_dir path/to/test/images \
  --output_csv results.csv
```

### Localization

Detect and localize forgeries in images:

```bash
python main.py \
  --mode localize \
  --input_dir path/to/test/images \
  --model_path data/models/rdlnn_model.pth \
  --localization_model_path data/models/pdywt_localizer.pth \
  --output_dir data/results
```

### Running the API Server

Start the API server for web-based forgery detection:

```bash
python api/api_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --detection-model data/models/rdlnn_model.pth \
  --localization-model data/models/pdywt_localizer.pth \
  --threshold 0.8
```

## API Endpoints

Once the server is running, the following endpoints are available:

- `GET /api/health`: Check server status
- `POST /api/detect`: Detect if an image is forged
- `POST /api/localize`: Detect and localize forgery in an image
- `POST /api/batch/detect`: Process multiple images for forgery detection
- `GET /api/results/latest`: Get the latest detection and localization results
- `GET /api/system/status`: Get detailed system status information

Example using curl:

```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/api/detect
```

## Frontend (Under Development)

A web-based frontend for the forgery detection system is currently under development. The frontend will provide an intuitive user interface for:

- Uploading and analyzing individual images
- Batch processing multiple images
- Visualizing forgery detection results
- Displaying localization heatmaps of potentially forged regions
- Viewing and comparing historical detection results

The frontend is being built with modern web technologies and will interact with the API server. Please note that this component is still in active development and may not be fully functional.

To run the frontend (when available):

```bash
cd frontend
npm install
npm start
```

This will start the development server, typically accessible at http://localhost:3000.

## Core Components

### 1. Feature Extraction

The system uses several feature extraction techniques:

- **PDyWT Features**: Polar Dyadic Wavelet Transform combines traditional wavelet decomposition with polar coordinates for rotation invariance
- **Error Level Analysis (ELA)**: Identifies inconsistencies in JPEG compression artifacts
- **Noise Analysis**: Detects inconsistencies in image noise patterns
- **DCT Features**: Analyzes Discrete Cosine Transform coefficients for manipulation detection
- **JPEG Ghost Analysis**: Detects traces of previous JPEG compressions that may indicate manipulation

### 2. Detection Model

The RDLNN (Regression Deep Learning Neural Network) architecture is specially designed to handle the imbalanced nature of forgery detection datasets, with:

- Batch normalization for stable training
- LeakyReLU activations for improved gradient flow
- Dropout for regularization
- Class weighting for handling imbalanced data
- Threshold optimization for precision-recall tradeoff

### 3. Localization Model

The localization model uses a simplified U-Net architecture to generate pixel-wise predictions of potentially forged regions.

## Directory Structure

```
├── api/                      # API server implementation
├── data/                     # Data directory (not tracked in git)
│   ├── features/             # Extracted features
│   ├── models/               # Trained models
│   └── results/              # Test results and visualizations
├── dwt/                      # Discrete Wavelet Transform implementation
│   ├── main.py               # DWT main entry point
│   ├── train.py              # DWT model training
│   ├── test.py               # DWT model testing
│   ├── utils.py              # DWT utility functions
│   └── report/               # DWT training reports
├── dywt/                     # Dyadic Wavelet Transform implementation
│   ├── main.py               # DyWT main entry point
│   ├── train.py              # DyWT model training
│   ├── test.py               # DyWT model testing
│   ├── utils.py              # DyWT utility functions
│   └── report/               # DyWT training reports
├── modules/                  # Core implementation modules
├── tools/                    # Utility scripts
├── training/                 # Training methods
├── main.py                   # Main entry point
└── requirements.txt          # Python dependencies
```

## Wavelet Transform Implementations

This system includes three different wavelet transform implementations:

### Discrete Wavelet Transform (DWT)

The DWT implementation uses a traditional 2D wavelet decomposition with Haar wavelets. It extracts statistical features from wavelet coefficients including:

- Mean, standard deviation, entropy, and energy of each subband
- Simple classification with Random Forest

Performance:

- Accuracy: ~66.55%
- Precision: 0.60
- Recall: 0.48
- F1-score: 0.53

### Dyadic Wavelet Transform (DyWT)

The DyWT implementation uses a multi-level wavelet decomposition with Daubechies wavelets (db4). It extracts:

- Statistical features (mean, std, skewness, kurtosis, entropy, energy)
- GLCM features (contrast, dissimilarity, homogeneity, energy, correlation)
- Supports multiple decomposition levels

Performance:

- Accuracy: ~80.47%
- Precision: 0.59
- Recall: 0.27
- F1-score: 0.37

### Polar Dyadic Wavelet Transform (PDyWT)

The PDyWT is the main implementation, combining wavelet decomposition with polar coordinate transformations to create rotation-invariant features. It includes:

- YCbCr color space processing
- Error Level Analysis (ELA)
- Noise features
- JPEG ghost detection
- DCT features

Performance metrics:

- ROC AUC: 0.968
- PR AUC: 0.908
- Accuracy: 96.17%
- Precision: 0.9290
- Recall: 0.9951
- F1 Score: 0.9609
- Specificity: 0.9232

The RDLNN model with PDyWT features provides exceptional performance with high confidence levels (45.6% of predictions in high-confidence category) and minimal uncertainty (only 8.5% of samples in the uncertain range), making it particularly suitable for real-world forgery detection applications where reliable classification is required.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
