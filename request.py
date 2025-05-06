import requests
import json
import os

# API endpoint URL - let's use the RDLNN-specific endpoint
url = "http://localhost:5000/api/rdlnn/detect"  # Use RDLNN-specific endpoint

# Path to the image file
image_path = r"D:\RDLNN with CNN-FE\data\CASIA2\Tp\Tp_D_CND_M_N_art00077_art00076_10290.tif"
model_path = r".\data\models\rdlnn_model.pth"
threshold = 0.675

# Prepare the files and form data for the request
files = {
    'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/tiff')
}

data = {
    'threshold': threshold
}

# Make the POST request to the API
response = requests.post(url, files=files, data=data)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    
    # Print the results
    print(f"Result: {result['result'].upper()}")
    print(f"Confidence: {result['probability']:.4f}")
    print(f"Threshold: {result['threshold']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
else:
    print(f"Error: {response.status_code}")
    print(response.text)