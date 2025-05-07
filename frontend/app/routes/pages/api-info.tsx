import { Link } from "react-router";

export default function Api() {
  return (
    <div className="container mx-auto px-4 pb-8 bg-gunmetal text-silversand">
      <Link
        to=".."
        className="sticky top-0 left-0 inline-flex items-center gap-1 text-sm font-medium text-neutral-600 hover:text-black bg-silversand transition-all duration-300 group p-2 pt-0.5 mx-2 rounded-bl-2xl rounded-br-2xl"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-3.5 w-3.5 opacity-70 group-hover:opacity-100 transform group-hover:-translate-x-0.5 transition-all duration-300"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z"
            clipRule="evenodd"
          />
        </svg>
        <span className="relative overflow-hidden">
          <span className="block">Go Back</span>
          <span className="absolute bottom-0 left-0 w-full h-px bg-black transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-300 ease-out"></span>
        </span>
      </Link>
      <h1 className="text-3xl font-bold mb-6">Image Forgery Detection API</h1>

      <div className="mb-6 border-l-4 border-silversand pl-4">
        <p className="mb-2 font-mono">
          # API Documentation # Image Forgery Detection System - v1.0 # RDLNN /
          DWT / DyWT Implementation
        </p>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Base Configuration
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`BASE_URL = "http://localhost:5000"
DEFAULT_THRESHOLD = 0.675
BATCH_SIZE = 32
DEFAULT_MODEL = "rdlnn_model.pth"
LOCALIZATION_MODEL = "pdywt_localizer.pth"`}
        </pre>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Health & Status Endpoints
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API server is running correctly"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': True,
        'upload_folder': '/path/to/upload',
        'threshold': 0.675
    })

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get detailed system status"""
    return jsonify({
        'status': 'healthy',
        'cuda': CUDA_STATUS,
        'models': MODELS_STATUS,
        'directories': DIRS_STATUS
    })

@app.route('/api/models/info', methods=['GET'])
def models_info():
    """Get information about available models"""
    return jsonify({
        'models': MODELS_INFO,
        'timestamp': datetime.now().isoformat()
    })`}
        </pre>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Detection Endpoints
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`@app.route('/api/rdlnn/detect', methods=['POST'])
def rdlnn_detect():
    """Detect forgery using RDLNN model
    
    Params:
        image: file - Image to analyze
        threshold: float - Optional detection threshold (0.0-1.0)
        
    Returns:
        detection_result: dict - Analysis results
    """
    # Implementation details
    return detection_result

@app.route('/api/dwt/detect', methods=['POST'])
def dwt_detect():
    """Detect forgery using Discrete Wavelet Transform model
    
    Params:
        image: file - Image to analyze
        threshold: float - Optional detection threshold (0.0-1.0)
        
    Returns:
        detection_result: dict - Analysis results
    """
    # Implementation details
    return detection_result

@app.route('/api/dywt/detect', methods=['POST'])
def dywt_detect():
    """Detect forgery using Dyadic Wavelet Transform model
    
    Params:
        image: file - Image to analyze
        threshold: float - Optional detection threshold (0.0-1.0)
        
    Returns:
        detection_result: dict - Analysis results
    """
    # Implementation details
    return detection_result`}
        </pre>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Localization Endpoints
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`@app.route('/api/localize', methods=['POST'])
def localize_forgery():
    """Detect and localize forgery in an image
    
    Params:
        image: file - Image to analyze
        
    Returns:
        localization_result: dict - Contains detection results and forgery map
    """
    # Implementation details
    return localization_result

@app.route('/api/dwt/localize', methods=['POST'])
def dwt_localize():
    """Localize forgery using DWT method
    
    Params:
        image: file - Image to analyze
        
    Returns:
        localization_result: dict - Contains detection results and forgery map
    """
    # Implementation details
    return localization_result

@app.route('/api/dywt/localize', methods=['POST'])
def dywt_localize():
    """Localize forgery using DyWT method
    
    Params:
        image: file - Image to analyze
        
    Returns:
        localization_result: dict - Contains detection results and forgery map
    """
    # Implementation details
    return localization_result`}
        </pre>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Advanced Analysis Endpoints
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`@app.route('/api/batch/detect', methods=['POST'])
def batch_detect():
    """Process multiple images for forgery detection
    
    Params:
        images[]: files - Multiple images to analyze
        
    Returns:
        batch_results: dict - Analysis results for all images
    """
    # Implementation details
    return batch_results

@app.route('/api/compare', methods=['POST'])
def compare_images():
    """Compare two images for similarity and potential forgery
    
    Params:
        image1: file - First image to compare
        image2: file - Second image to compare
        
    Returns:
        comparison_result: dict - Comparison results with similarity score
    """
    # Implementation details
    return comparison_result

@app.route('/api/analyze/threshold', methods=['POST'])
def analyze_threshold():
    """Analyze the effect of different thresholds on prediction results
    
    Params:
        image: file - Image to analyze
        
    Returns:
        threshold_results: dict - Results for different threshold values
    """
    # Implementation details
    return threshold_results`}
        </pre>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Response Formats
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`# Detection Response Format
detection_result = {
    'filename': str,          # Original filename
    'result': str,            # 'forged' or 'authentic'
    'probability': float,     # Confidence score from 0-1
    'threshold': float,       # Detection threshold
    'processing_time': float, # Seconds
    'timestamp': int,         # Unix timestamp
    'original_image_base64': str,  # Base64 encoded image
    'result_text': str        # Human-readable result
}

# Localization Response Format
localization_result = {
    **detection_result,       # All detection fields plus:
    'forgery_map_base64': str,  # Base64 encoded heatmap
    'regions': [              # Detected forgery regions
        {
            'x': int,         # X coordinate
            'y': int,         # Y coordinate
            'width': int,     # Width of region
            'height': int     # Height of region
        },
        # More regions...
    ]
}

# Error Response Format
error_response = {
    'error': str,           # Error message
    'traceback': str        # Detailed error info (dev mode only)
}`}
        </pre>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Usage Examples
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`# Python Example
import requests

def detect_forgery(image_path):
    """Detect if an image is forged using RDLNN model
    
    Args:
        image_path: str - Path to image file
        
    Returns:
        result: dict - Detection results
    """
    url = 'http://localhost:5000/api/rdlnn/detect'
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.text}")

# JavaScript/Fetch Example
async function detectForgery(imageFile) {
  const url = 'http://localhost:5000/api/rdlnn/detect';
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(url, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  
  return await response.json();
}`}
        </pre>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-3 font-mono">
          # Model Architecture
        </h2>
        <pre className="bg-black bg-opacity-40 p-4 rounded-lg mb-4 overflow-x-auto text-sm">
          {`class RegressionDLNN:
    """Deep Learning Neural Network for image forgery detection
    
    Uses regression instead of classification for better performance
    on forgery detection tasks.
    """
    
    def __init__(self, input_dim=1024, hidden_layers=[512, 256, 128]):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.model = self._build_model()
        self.threshold = 0.675  # Default threshold
        
    def _build_model(self):
        """Build the neural network model architecture"""
        layers = []
        # Model architecture details...
        return nn.Sequential(*layers)
        
    @classmethod
    def load(cls, model_path):
        """Load model from file"""
        # Implementation details
        
    def predict(self, features):
        """Make forgery prediction based on features
        
        Args:
            features: Extracted features from image
            
        Returns:
            prediction: Binary forgery prediction (0/1)
            probability: Confidence score (0-1)
        """
        # Implementation details

class PDyWTCNNDetector:
    """Image forgery detector that combines PDyWT features with CNN
    
    Uses wavelet transform for feature extraction and CNN for
    forgery detection and localization.
    """
    
    def __init__(self, model_path, localization_model_path):
        self.rdlnn_model = RegressionDLNN.load(model_path)
        self.localization_model = self._load_localization_model(
            localization_model_path)
        self.threshold = self.rdlnn_model.threshold
        
    def detect(self, image_path):
        """Detect if an image is forged
        
        Args:
            image_path: Path to image file
            
        Returns:
            result: Dict with prediction and probability
        """
        # Implementation details
        
    def localize(self, image_path, save_path=None):
        """Detect and localize forgery in an image
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save forgery map
            
        Returns:
            result: Dict with prediction, probability and
                   forgery map
        """
        # Implementation details`}
        </pre>
      </div>

      <div className="mt-10 pt-6 border-t border-silversand border-opacity-40 text-center text-sm">
        <p>Image Forgery Detection System - API v1.0</p>
        <p className="text-xs opacity-70 mt-1">
          Combined RDLNN, DWT and DyWT implementations for forgery detection and
          localization
        </p>
      </div>
    </div>
  );
}
