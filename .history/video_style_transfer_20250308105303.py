import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from model import TransformerNet

def load_model(model_path, device):
    model = TransformerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def stylize_frame(model, frame, device):
    # Convert frame from BGR (OpenCV) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image and apply transformations
    pil_image = Image.fromarray(frame)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    # Transform image
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Forward pass through model
    with torch.no_grad():
        output = model(image_tensor)
    
    # Post-process output
    output = output[0].cpu().clone()
    output = output.clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output

def main():
    # Set device (MPS for Apple Silicon, CPU otherwise)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Available styles (example)
    style_models = {
        '1': 'models/starry_night.pth',
        '2': 'models/mosaic.pth',
        '3': 'models/udnie.pth',
        # Add more if available
    }
    
    # List available styles
    print("Available styles:")
    for key, value in style_models.items():
        style_name = os.path.basename(value).split('.')[0]
        print(f"{key}: {style_name}")
    
    # Get style selection
    style_key = input("Select a style (1-3): ")
    
    # Load model
    if style_key in style_models:
        model_path = style_models[style_key]
        model = load_model(model_path, device)
    else:
        print("Invalid selection, using default style")
        model = load_model(style_models['1'], device)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    
    # FPS calculation variables
    prev_time = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Resize frame for better performance
        frame = cv2.resize(frame, (320, 240))
        
        # Apply style transfer
        output = stylize_frame(model, frame, device)
        
        # Calculate FPS
        fps_count += 1
        if fps_count >= 10:
            curr_time = time.time()
            fps = fps_count / (curr_time - prev_time)
            fps_count = 0
            prev_time = curr_time
        
        # Display FPS
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frames
        cv2.imshow('Original', frame)
        cv2.imshow('Stylized', output)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()