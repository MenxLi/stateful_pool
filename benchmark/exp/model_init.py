import torch
from torchvision.models import vit_l_16
from torchvision import transforms

def initialize_model(device_str: str):
    print(f"Initializing ViT-L on {device_str} with random weights (fixed seed)...")
    device = torch.device(device_str)
    
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize ViT-L without downloading weights (random init)
    model = vit_l_16(weights=None)
    model.to(device)
    model.eval()
    
    # Standard Transform for ViT
    # Using 224x224 input size primarily
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, preprocess