import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
import os
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define possible paths for the model
PATHS_TO_TRY = [
    # First try same directory as script (root/main directory)
    os.path.join(SCRIPT_DIR, 'efficientnet_b0_medium_augmentation.pth'),
    # Try just the filename in case SCRIPT_DIR is different
    'efficientnet_b0_medium_augmentation.pth',
    # Try models folder in case you move it later
    os.path.join(SCRIPT_DIR, 'models', 'efficientnet_b0_medium_augmentation.pth'),
]

# Find the actual model path
MODEL_PATH = None
for path in PATHS_TO_TRY:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    print("❌ Model file not found in any of the expected locations:")
    for path in PATHS_TO_TRY:
        print(f"  - {path}")
    MODEL_PATH = PATHS_TO_TRY[0]  # Use first path as fallback

print(f"📁 Model path: {MODEL_PATH}")

# Model configuration for EfficientNet B0
MODEL_CONFIG = {
    'model_fn': models.efficientnet_b0,
    'feature_dim': 1280,
    'input_size': 224
}

def create_model(num_classes, pretrained=False):
    """Create EfficientNet B0 model with custom classifier"""
    model = MODEL_CONFIG['model_fn'](pretrained=pretrained)
    model.classifier[1] = nn.Linear(MODEL_CONFIG['feature_dim'], num_classes)
    return model

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def load_model():
    """Load the trained model"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        num_classes = 2  # Adjust based on your dataset
        model = create_model(num_classes, pretrained=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict(image):
    """Make prediction on uploaded image"""
    try:
        if model is None:
            return {"Error": "Model not loaded properly"}
            
        transform = get_transforms()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        probabilities = probabilities.cpu().numpy()[0]
        class_names = ['negatif', 'positif']
        
        return {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}

# Load model
model = load_model()

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=2, label="Predictions"),
    title="🤖 EfficientNet B0 Image Classifier",
    description="Upload an image to get predictions with confidence scores.",
    allow_flagging="never"
)

# For Vercel deployment, we need to export the app
app = iface

if __name__ == "__main__":
    # For local development
    iface.launch(server_name="0.0.0.0", server_port=8080)
