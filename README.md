# Image Forgery Detection System

This repository contains a comprehensive framework for detecting and localizing image forgeries using deep learning and wavelet-based feature extraction techniques. The system leverages Polar Dyadic Wavelet Transform (PDyWT) with Convolutional Neural Networks (CNNs) and Regression Deep Learning Neural Networks (RDLNN) to identify manipulated images.

## Features

- **Image Forgery Detection**: Classify images as authentic or forged with high precision
- **Forgery Localization**: Identify and highlight specific regions of an image that have been tampered with
- **PDyWT Feature Extraction**: Extract robust wavelet-based features that are effective for forgery detection
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

## Architecture

### PDyWT Feature Extraction

The Polar Dyadic Wavelet Transform (PDyWT) combines traditional wavelet decomposition with polar coordinate transformations to create rotation-invariant features. These features are particularly effective for detecting image manipulation.

Key components:
- Wavelet decomposition using Haar wavelets
- Polar coordinate transformation for rotation invariance
- Feature vector extraction using average pooling

### RDLNN Model

The Regression Deep Learning Neural Network (RDLNN) is specially designed to handle the imbalanced nature of forgery detection datasets. It includes:

- Batch normalization layers for stable training
- LeakyReLU activations for improved gradient flow
- Dropout for regularization
- Class weighting for handling imbalanced data
- Threshold optimization for precision-recall tradeoff

### Forgery Localization

The localization model uses a simplified U-Net architecture to generate pixel-wise predictions of potentially forged regions.

## Project Structure

```
├── api/                      # API server implementation
├── data/                     # Data directory (not tracked in git)
│   ├── features/             # Extracted features
│   ├── models/               # Trained models
│   └── results/              # Test results and visualizations
├── modules/                  # Core implementation modules
│   ├── batch_processor.py    # Batch processing utilities
│   ├── data_handling.py      # Data loading and processing
│   ├── feature_extractor.py  # PDyWT feature extraction
│   ├── model.py              # Model definitions
│   ├── preprocessing.py      # Image preprocessing
│   ├── rdlnn.py              # RDLNN implementation
│   └── utils.py              # Utility functions
├── tools/                    # Utility scripts
│   ├── combine_features.py   # Feature combination
│   ├── extract_pdywt_features.py  # Feature extraction script
│   ├── train_localization.py      # Localization model training
│   └── train_pdywt_models.py      # PDyWT model training
├── training/                 # Training methods
│   ├── balanced.py           # Balanced training approaches
│   └── precision.py          # Precision-tuned training
├── main.py                   # Main entry point
└── requirements.txt          # Python dependencies
```

## Performance Tips

- Use GPU acceleration for training and feature extraction when available
- Pre-extract features for large datasets to speed up training
- Use batch processing for efficient evaluation of multiple images
- Set an appropriate threshold (typically 0.7-0.8) to balance precision and recall
- For large-scale deployment, consider using the API server with a front-end interface

## Citation

If you use this project in your research, please cite:

```
@software{image_forgery_detection,
  author = {Your Name},
  title = {Image Forgery Detection System},
  year = {2023},
  url = {https://github.com/yourusername/image-forgery-detection}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
