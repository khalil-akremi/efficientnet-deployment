import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
import os
import numpy as np

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATHS_TO_TRY = [
    # First try relative path
    os.path.join(SCRIPT_DIR, 'models', 'efficientnet_b0_medium_augmentation.pth'),
    # Then try absolute path
    r'C:\Users\user\Desktop\efficientnet-deployment\model\efficientnet_b0_medium_augmentation.pth',
    # Add other possible locations if needed
]
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

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080)