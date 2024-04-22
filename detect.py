# Load model directly
from PIL import Image
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load processor and model on the appropriate device
processor = AutoImageProcessor.from_pretrained("vit-fire-detection")
model = AutoModelForImageClassification.from_pretrained("vit-fire-detection").to(device)


def make_inference(image_path):
    # Load and preprocess image
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(device)
              
    # Perform inference
    outputs = model(**inputs)
                
    # Get predicted probabilities
    predicted_probabilities = outputs.logits.softmax(dim=-1)
                
    # Assuming the first class is negative and the second class is positive
    positive_probability = predicted_probabilities[0][1].item()  # Probability of positive class
    negative_probability = predicted_probabilities[0][0].item()  # Probability of negative class
                
    # Determine predicted class based on probabilities
    return "No fire" if positive_probability > negative_probability else "Fire"