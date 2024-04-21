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

# Directory containing images
images_folder = "Fire_detection_Data/fire"

# List to store predictions
all_predictions = []
problematic_images = []

count = 0
all_count = 0
# Iterate over images in the folder

for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are in JPG or PNG format
            image_path = os.path.join(images_folder, filename)
            try:
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
                predicted_class = "No fire" if positive_probability > negative_probability else "Fire"
                if predicted_class == "No fire":
                    count = count+1
                all_count = all_count+1
                # Append prediction to the list
                all_predictions.append({"image_path": image_path, "predicted_class": predicted_class, "positive_probability": positive_probability, "negative_probability": negative_probability})
            except Exception as e:
                problematic_images.append({"image_path": image_path, "error": str(e)})


'''
# Print predictions
for prediction in all_predictions:
    if prediction['predicted_class'] == "Fire":
        print(f"Image: {prediction['image_path']}")
        print(f"Image: {prediction['negative_probability']}")
        print("--------------------")
'''

print((all_count-count)/all_count)