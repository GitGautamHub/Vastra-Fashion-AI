

import pandas as pd
import requests
from tqdm.notebook import tqdm 
import os
from PIL import Image 
import ast 
# --- Configuration ---

CSV_FILES = [
    os.path.join('data', 'dresses_bd_processed_data.csv'),
    os.path.join('data', 'jeans_bd_processed_data.csv')
]
IMAGES_DIR = 'downloaded_fashion_images' 
OUTPUT_PROCESSED_DATA_PATH = os.path.join('data', 'vastra_processed_data_with_local_paths.csv') # Combined output file


os.makedirs('data', exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"Ensured '{os.path.join('data')}' and '{IMAGES_DIR}' directories exist.")

# --- Function to download an image ---
def download_image(url, save_path, product_id):
    """Downloads an image from a URL and saves it to a specified path."""
    if pd.isna(url) or url == '': # Check if URL is NaN or empty string
        return None
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        with open(save_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image for product_id {product_id} from {url}: {e}", flush=True)
        return None
    except Exception as e:
        print(f"An unexpected error occurred for product_id {product_id} from {url}: {e}", flush=True)
        return None

# --- Main execution block ---
if __name__ == "__main__":
    print(f"Attempting to load datasets from: {CSV_FILES}")
    
    all_dfs = []
    for file_path in CSV_FILES:
        try:
            current_df = pd.read_csv(file_path)
            all_dfs.append(current_df)
            print(f"Loaded {len(current_df)} rows from {file_path}")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {file_path}.")
            print(f"Please ensure '{os.path.basename(file_path)}' is placed inside the 'data' folder.")
            # Depending on your requirement, you might want to exit here or continue with other files.
            # For now, we'll continue to try loading other files.
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    if not all_dfs:
        print("No datasets were loaded. Exiting.")
        exit()

    # Concatenate all dataframes into a single one
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nSuccessfully combined {len(all_dfs)} datasets.")
    print(f"Total rows in combined DataFrame: {len(df)}")
    
    print("\nCombined DataFrame Info:")
    df.info()
    print("\nFirst 5 rows of the combined dataset:")
    print(df.head())
    print("\nMissing values for key image columns (feature_image_s3):")
    print(df['feature_image_s3'].isnull().sum())
    
    
    if 'product_id' in df.columns:
        df['product_id'] = df['product_id'].astype(str)
    else:
        print("Warning: 'product_id' column not found. Generating unique IDs for image names.")
        df['product_id'] = [f"item_{i}" for i in range(len(df))] # Fallback unique ID generation


    initial_rows = len(df)
    df.drop_duplicates(subset=['product_id'], inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate product IDs.")


    # --- Download all feature images ---
    print(f"\nStarting primary feature image download to {IMAGES_DIR}...")
    downloaded_image_paths = []
    skipped_downloads = 0


    for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Images"):
        product_id = row['product_id']
        image_url = row['feature_image_s3']

        if pd.isna(image_url) or image_url == '':
            skipped_downloads += 1
            downloaded_image_paths.append(None)
            continue

        try:
            # Try to infer extension, default to .jpg if not found
            file_extension = os.path.splitext(image_url)[1]
            if not file_extension or len(file_extension) > 5 or '?' in file_extension:
                file_extension = '.jpg' # Default to jpg if extension looks weird or is too long
            
            image_name = f"{product_id}{file_extension.lower()}"
            save_path = os.path.join(IMAGES_DIR, image_name)

            if not os.path.exists(save_path): # Download only if not already downloaded
                path = download_image(image_url, save_path, product_id)
                if path:
                    downloaded_image_paths.append(path)
                else:
                    downloaded_image_paths.append(None)
                    skipped_downloads += 1
            else:
                downloaded_image_paths.append(save_path) # Already exists, just add path

        except Exception as e:
            print(f"Error processing URL {image_url} for product_id {product_id}: {e}", flush=True)
            downloaded_image_paths.append(None)
            skipped_downloads += 1


    df['local_image_path'] = downloaded_image_paths

    print(f"\nImage download complete. Total images processed: {len(df)}")
    successful_downloads = len(df) - df['local_image_path'].isnull().sum()
    print(f"Successfully tracked image paths for: {successful_downloads} images.")
    print(f"Skipped/Failed downloads (due to missing URL, download error, etc.): {skipped_downloads} images.")


    df_processed = df.dropna(subset=['local_image_path']).copy()
    print(f"DataFrame after filtering out products with no local image: {len(df_processed)} rows.")

    # Save the updated DataFrame with local paths
    df_processed.to_csv(OUTPUT_PROCESSED_DATA_PATH, index=False)
    print(f"\nUpdated and combined DataFrame saved to '{OUTPUT_PROCESSED_DATA_PATH}'")


    try:
        import matplotlib.pyplot as plt
        import random

        print("\nDisplaying a few random downloaded images (requires matplotlib)...")
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        valid_sample_paths = [path for path in df_processed['local_image_path'].sample(min(5, len(df_processed))).tolist() if path and os.path.exists(path)]

        if valid_sample_paths:
            for i, img_path in enumerate(valid_sample_paths):
                try:
                    img = Image.open(img_path).convert('RGB')
                    axes[i].imshow(img)
                    axes[i].set_title(os.path.basename(img_path)[:10] + '...') # Show first 10 chars of filename
                    axes[i].axis('off')
                except Exception as e:
                    axes[i].set_title(f"Error loading: {e}")
                    axes[i].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("No valid images to display yet.")

    except ImportError:
        print("\nMatplotlib not installed. Skipping image display. (Install with: pip install matplotlib)")
    except Exception as e:
        print(f"\nError displaying images: {e}")