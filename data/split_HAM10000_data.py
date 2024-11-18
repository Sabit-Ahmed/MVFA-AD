import pandas as pd
import os

from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

# Paths
dataset_path = "Skin_HAM10000_AD/preprocessing/"  # Root directory of the dataset
images_path = os.path.join(dataset_path, "HAM10000_images/")  # Directory containing images
masks_path = os.path.join(dataset_path, "HAM10000_segmentations_masks/")
output_path = "/data/Skin_HAM10000_AD"  # Directory to save split data
csv_path = os.path.join(dataset_path, "HAM10000_metadata.csv")  # Metadata CSV

label_path = {"MEL": "good", "NV": "Ungood"}
# Create output directories for splits
splits = ["train", "valid", "test"]
for split in splits:
    for label in ["MEL", "NV"]:
        os.makedirs(os.path.join(output_path, split, label_path[label], 'img'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, label_path[label], 'anomaly_mask'), exist_ok=True)

# Load metadata
data = pd.read_csv(csv_path)

# Ensure required columns exist
required_columns = ["image_id", "dx",]
if not all(column in data.columns for column in required_columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

# Filter only MEL and NV classes
data = data[data["dx"].isin(["mel", "nv"])]

# Map dx column to MEL and NV labels
data["dx"] = data["dx"].map({"mel": "MEL", "nv": "NV"})

# Train-Test-Validation split (80% train, 10% validation, 10% test)
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data["dx"], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["dx"], random_state=42)

# Function to copy images and masks
def copy_files(dataframe, source_images, source_masks, output_dir):
    for _, row in dataframe.iterrows():
        image_filename = row["image_id"] + ".jpg"
        mask_filename = row["image_id"] + "_segmentation" + ".png"
        dest_mask_name = row["image_id"] + ".png"
        label = row["dx"]

        # Copy image for classification
        src_image_path = os.path.join(source_images, image_filename)
        if os.path.exists(src_image_path):
            with Image.open(src_image_path) as img:
                # Convert to PNG
                new_image_name = f"{row['image_id']}.png"
                dst_image_path = os.path.join(output_dir, label_path[label], "img", new_image_name)
                img.save(dst_image_path, "PNG")
                # print(f"Converted and saved: {new_image_name}")
        else:
            print(f"Image not found: {src_image_path}")

        # Copy mask for segmentation
        src_mask_path = os.path.join(source_masks, mask_filename)
        if os.path.exists(src_mask_path):  # Ensure the mask exists
            dst_mask_path = os.path.join(output_dir, label_path[label], 'anomaly_mask', dest_mask_name)
            shutil.copy(src_mask_path, dst_mask_path)
        else:
            print(f"Mask not found for {mask_filename}. Skipping.")

# Copy files for each split
for split, split_data in zip(splits, [train_data, val_data, test_data]):
    split_output_path = os.path.join(output_path, split)
    copy_files(split_data, images_path, masks_path, split_output_path)

print("Dataset splitting completed!")
