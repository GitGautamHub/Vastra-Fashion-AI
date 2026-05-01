import pandas as pd
import requests
from tqdm import tqdm 
import os
from PIL import Image 
import ast 
from concurrent.futures import ThreadPoolExecutor, as_completed 


# --- Configuration ---
CSV_FILES = [
    os.path.join('data', 'dresses_bd_processed_data.csv'),
    os.path.join('data', 'jeans_bd_processed_data.csv')
]
IMAGES_DIR = 'downloaded_fashion_images' 
OUTPUT_PROCESSED_DATA_PATH = os.path.join('data', 'vastra_processed_data_with_local_paths.csv') # Combined output file


MAX_WORKERS = 10 # You can adjust this based on your internet speed and CPU


os.makedirs('data', exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"Ensured '{os.path.join('data')}' and '{IMAGES_DIR}' directories exist.")

# --- Function to download an image ---
# NOTE: This function is slightly modified to return the product_id and path
def download_image(url, save_path, product_id):
    """Downloads an image from a URL and saves it to a specified path."""
    if pd.isna(url) or url == '': # Check if URL is NaN or empty string
        return product_id, None

    if os.path.exists(save_path):
        return product_id, save_path # Skip download if file already exists

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        with open(save_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        return product_id, save_path
    except requests.exceptions.RequestException as e:
        # Use print here as it's running in a thread, flush=True helps
        print(f"Error downloading image for product_id {product_id} from {url}: {e}", flush=True) 
        return product_id, None
    except Exception as e:
        print(f"An unexpected error occurred for product_id {product_id} from {url}: {e}", flush=True)
        return product_id, None

# --- Main execution block ---
if __name__ == "__main__":
    print(f"Attempting to load datasets from: {CSV_FILES}")
    
    # ... (DataFrame loading, concatenation, and deduplication logic remains the same) ...
    all_dfs = []
    for file_path in CSV_FILES:
        try:
            current_df = pd.read_csv(file_path)
            all_dfs.append(current_df)
            print(f"Loaded {len(current_df)} rows from {file_path}")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {file_path}.")
            print(f"Please ensure '{os.path.basename(file_path)}' is placed inside the 'data' folder.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    if not all_dfs:
        print("No datasets were loaded. Exiting.")
        exit()

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nSuccessfully combined {len(all_dfs)} datasets.")
    print(f"Total rows in combined DataFrame: {len(df)}")
    
    if 'product_id' in df.columns:
        df['product_id'] = df['product_id'].astype(str)
    else:
        df['product_id'] = [f"item_{i}" for i in range(len(df))]

    initial_rows = len(df)
    df.drop_duplicates(subset=['product_id'], inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate product IDs.")

    # --- Prepare download arguments and Filter out missing URLs before parallelizing ---
    download_args = []
    for index, row in df.iterrows():
        product_id = row['product_id']
        image_url = row['feature_image_s3']

        if pd.isna(image_url) or image_url == '':
            continue
            
        # Try to infer extension, default to .jpg if not found
        file_extension = os.path.splitext(image_url)[1]
        if not file_extension or len(file_extension) > 5 or '?' in file_extension:
            file_extension = '.jpg'
            
        image_name = f"{product_id}{file_extension.lower()}"
        save_path = os.path.join(IMAGES_DIR, image_name)
        
        # Store the arguments
        download_args.append((image_url, save_path, product_id))

    print(f"\nStarting primary feature image download to {IMAGES_DIR} with {MAX_WORKERS} parallel threads...")
    
    # --- PARALLEL DOWNLOAD LOGIC ---
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_product = {
            executor.submit(download_image, url, path, pid): pid 
            for url, path, pid in download_args
        }

        # Use tqdm to monitor the progress of completed tasks
        for future in tqdm(as_completed(future_to_product), total=len(future_to_product), desc="Downloading Images"):
            product_id, local_path = future.result()
            results[product_id] = local_path
    
    # Map results back to the DataFrame
    df['local_image_path'] = df['product_id'].map(results)
    # -------------------------------
    
    # ... (The rest of the post-download summary and saving logic remains the same) ...

    print(f"\nImage download complete. Total products considered: {len(df)}")
    successful_downloads = df['local_image_path'].count() # Count non-None values
    skipped_downloads = len(df) - successful_downloads
    print(f"Successfully tracked image paths for: {successful_downloads} images.")
    print(f"Skipped/Failed downloads (due to missing URL, download error, etc.): {skipped_downloads} images.")

    df_processed = df.dropna(subset=['local_image_path']).copy()
    print(f"DataFrame after filtering out products with no local image: {len(df_processed)} rows.")

    # Save the updated DataFrame with local paths
    df_processed.to_csv(OUTPUT_PROCESSED_DATA_PATH, index=False)
    print(f"\nUpdated and combined DataFrame saved to '{OUTPUT_PROCESSED_DATA_PATH}'")

    # ... (Matplotlib display logic) ...
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