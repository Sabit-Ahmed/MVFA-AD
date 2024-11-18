import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

# Paths
csv_file_path = 'Skin_ISIC2019_AD/preprocessing/ISIC_2019_Training_GroundTruth.csv'  # Replace with the path to your CSV file
images_folder_path = '/data/Skin_ISIC2019_AD/preprocessing/ISIC_2019_Training_Input/'  # Replace with the path to your images folder
output_folder = '/Users/bcw3zj/PycharmProjects/MVFA-AD/data/Skin_ISIC2019_AD/'  # Replace with the desired output folder path

# Output folders for splits
train_folder = os.path.join(output_folder, 'train')
val_folder = os.path.join(output_folder, 'valid')
test_folder = os.path.join(output_folder, 'test')

# Create output directories
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(os.path.join(folder, 'good', 'img'), exist_ok=True) # MEL
    os.makedirs(os.path.join(folder, 'Ungood', 'img'), exist_ok=True) # NV

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Ensure the CSV contains 'image', 'MEL', and 'NV' columns
if 'image' not in data.columns or 'MEL' not in data.columns or 'NV' not in data.columns:
    raise ValueError("The CSV file must contain 'image', 'MEL', and 'NV' columns.")

# Keep only 'image', 'MEL', and 'NV' columns
filtered_data = data[['image', 'MEL', 'NV']]

# Convert multi-label columns to a single label column
# The label will be either 'MEL' or 'NV' based on which column has a value of 1
filtered_data = filtered_data.melt(id_vars=['image'], var_name='label', value_name='value')
filtered_data = filtered_data[filtered_data['value'] == 1].drop(columns=['value'])

# First split: Train (80%) and temp (20%)
train_data, temp_data = train_test_split(
    filtered_data, test_size=0.2, stratify=filtered_data['label'], random_state=42
)

# Second split: Validation (10%) and Test (10%) from temp
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42
)

# Function to copy images to their respective directories
def copy_images(dataframe, source_folder, destination_folder):
    for _, row in dataframe.iterrows():
        image_name = row['image']
        label = row['label']
        label_path = 'good'
        if str(label) == 'NV':
            label_path = 'Ungood'
        label_folder = os.path.join(destination_folder, label_path, 'img')
        os.makedirs(label_folder, exist_ok=True)
        source_image_path = os.path.join(source_folder, image_name + '.jpg')  # Assumes images have .jpg extension
        destination_image_path = os.path.join(label_folder, image_name + '.jpg')
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, destination_image_path)
        else:
            print(f"Image {source_image_path} not found.")

# Copy images to respective folders
copy_images(train_data, images_folder_path, train_folder)
copy_images(val_data, images_folder_path, val_folder)
copy_images(test_data, images_folder_path, test_folder)

print("Dataset split completed!")