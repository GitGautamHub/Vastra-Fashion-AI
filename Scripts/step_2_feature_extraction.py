

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm # Use tqdm for Jupyter/Colab, or from tqdm import tqdm for console

# --- Configuration ---

PROCESSED_DATA_PATH = os.path.join('data', 'vastra_processed_data_with_local_paths.csv')
IMAGES_DIR = 'downloaded_fashion_images'
EMBEDDINGS_FILE = os.path.join('models', 'vastra_image_embeddings.npy')
PRODUCT_IDS_FILE = os.path.join('models', 'vastra_product_ids_for_embeddings.npy')


os.makedirs('models', exist_ok=True)
print(f"Ensured '{os.path.join('models')}' directory exists.")
print("Loading pre-trained ResNet50 model...")

# Use pre-trained ResNet50 with ImageNet weights
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval() 

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded and moved to device: {device}")

preprocess = transforms.Compose([
    transforms.Resize(256),         # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),     # Crop the central 224x224 pixels
    transforms.ToTensor(),          # Convert PIL Image to PyTorch Tensor (scales pixel values to [0,1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using ImageNet's mean and std
])

# --- Function to Extract Features from a Single Image ---
def extract_features(image_path, model, preprocess, device):
    
    try:
        img = Image.open(image_path).convert('RGB') 
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)
        
        features = features.squeeze().cpu().numpy()
        return features
    except Exception as e:
        return None

# --- Main execution block ---
if __name__ == "__main__":
    print(f"Loading processed data from: {PROCESSED_DATA_PATH}")
    try:
        # Load the combined processed data from Step 1
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Loaded {len(df_processed)} products with local image paths for feature extraction.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}.")
        print("Please run scripts/step_1_data_prep.py first to generate this combined file.")
        exit()
    except Exception as e:
        print(f"Error loading processed data file: {e}")
        exit()

    all_embeddings = []
    corresponding_product_ids = []
    skipped_count = 0

    print("\nStarting feature extraction for all valid images...")
    # Iterate through each row in the DataFrame (which contains local image paths)
    for index, row in tqdm(df_processed.iterrows(), total=len(df_processed), desc="Extracting Features"):
        local_image_path = row['local_image_path']
        product_id = row['product_id']

        # Ensure the image file actually exists before trying to process it
        if os.path.exists(local_image_path):
            features = extract_features(local_image_path, model, preprocess, device)
            if features is not None:
                all_embeddings.append(features)
                corresponding_product_ids.append(product_id)
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"\nFeature extraction complete. Total entries processed from DataFrame: {len(df_processed)}")
    print(f"Successfully extracted features for: {len(all_embeddings)} images.")
    print(f"Skipped/Failed feature extractions: {skipped_count} images.")

    if len(all_embeddings) == 0:
        print("No embeddings were extracted. Please check previous steps and image paths.")
        exit()

    # Convert list of embeddings to a single NumPy array (Faiss prefers float32)
    all_embeddings_np = np.array(all_embeddings).astype('float32')
    corresponding_product_ids_np = np.array(corresponding_product_ids)

    # Save the embeddings and corresponding product IDs to the 'models' folder
    np.save(EMBEDDINGS_FILE, all_embeddings_np)
    np.save(PRODUCT_IDS_FILE, corresponding_product_ids_np)

    print(f"\nImage embeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Corresponding product IDs saved to: {PRODUCT_IDS_FILE}")
    print(f"Shape of saved embeddings: {all_embeddings_np.shape}")