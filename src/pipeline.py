import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def get_preprocessing_transforms(image_size=(224, 224)):
    """
    Standar preprocessing pipeline for medical imaging (resizing, tensor conversion, normalization).
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # standard ImageNet means
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_medical_image(pil_image, image_size=(224, 224)):
    """
    Take PIL Image and return torch tensor ready for model input.
    """
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    transform = get_preprocessing_transforms(image_size)
    tensor = transform(pil_image).unsqueeze(0) # Add batch dimension: [1, 3, H, W]
    return tensor

def generate_mock_heatmap(image_size=(224, 224)):
    """
    Generates a localized Grad-CAM style heatmap mask for demonstration purposes.
    """
    h, w = image_size
    y, x = np.ogrid[:h, :w]
    # Center a gaussian circle representing tumor location
    cx, cy = int(w * 0.55), int(h * 0.45)
    radius = min(h, w) * 0.18
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = np.exp(- (dist_from_center**2) / (2 * (radius**2)))
    return (mask * 255).astype(np.uint8)
