import torch
import pandas as pd
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import os
import numpy as np

# Device check and model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Adjust paths to point to your local pretrained model directories
feature_extractor = SegformerFeatureExtractor.from_pretrained('path_to_segformer_feature_extractor')
model = SegformerForSemanticSegmentation.from_pretrained('path_to_segformer_model')
model.to(device).eval()

# Mapping from class indices to labels
id2label = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
}

# Define the selected classes
selected_classes = [
    "sidewalk", "building", "wall", "fence", "pole", 
    "traffic light", "traffic sign", "vegetation", "terrain"
]

def get_logits(image_path, feature_extractor, model, device):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return logits.squeeze(0).cpu()

def save_logits_with_pixels_to_csv(logits, image_path, selected_classes, id2label, new_base_dir, main_folder):
    # Flatten the logits
    logits_flat = logits.view(logits.shape[0], -1).t()

    # Convert class indices to labels for each pixel
    predicted_classes = logits.argmax(dim=0).view(-1).numpy()
    class_labels = [id2label[c] for c in predicted_classes]

    # Load the original image and flatten its pixel values
    original_image = Image.open(image_path).convert('RGB')
    image_flat = np.array(original_image).reshape(-1, 3)

    # Create a DataFrame for logits, naming the columns according to class tags
    logits_df = pd.DataFrame(logits_flat.numpy(), columns=[id2label[i] for i in range(logits.shape[0])])

    # Add a column for class labels
    logits_df['class_label'] = class_labels

    # Filter DataFrame to include only the selected classes and corresponding logits
    relevant_logits_df = logits_df[selected_classes + ['class_label']]
    relevant_logits_df = relevant_logits_df[relevant_logits_df['class_label'].isin(selected_classes)]

    # Create a DataFrame for pixel values
    pixels_df = pd.DataFrame(image_flat, columns=['pixel_r', 'pixel_g', 'pixel_b'])
    pixels_df = pixels_df.loc[relevant_logits_df.index]  # matching pixels

    # Combine pixel values and logits
    combined_df = pd.concat([pixels_df, relevant_logits_df], axis=1)

    # Construct new CSV path
    relative_path = os.path.relpath(image_path, start=os.path.commonpath([main_folder, image_path]))
    new_csv_path = os.path.join(new_base_dir, os.path.splitext(relative_path)[0] + '_logits_pixels.csv')
    os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)
    combined_df.to_csv(new_csv_path, index=False)

    return new_csv_path


def process_images_in_folder(main_folder, selected_classes, id2label, new_base_dir):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}")

                logits = get_logits(image_path, feature_extractor, model, device)
                csv_path = save_logits_with_pixels_to_csv(
                    logits, image_path, selected_classes, id2label, new_base_dir, main_folder
                )
                print(f"Saved logits and pixels to CSV at {csv_path}")


if __name__ == "__main__":
    # Adjust these paths as needed
    main_folder = 'path_to_images'
    new_base_dir = 'path_to_output_csvs'
    process_images_in_folder(main_folder, selected_classes, id2label, new_base_dir)
