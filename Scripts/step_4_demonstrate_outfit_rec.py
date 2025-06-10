

import os
import sys

# --- DEBUGGING PATH ---
print(f"Current working directory (os.getcwd()): {os.getcwd()}")
print(f"File's directory (os.path.dirname(__file__)): {os.path.dirname(__file__)}")

# Calculate the project root (Vastra folder)
# This assumes the script is in 'Scripts' folder, one level below the root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Calculated project_root: {project_root}")

# Add the project root to Python's system path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added '{project_root}' to sys.path.")
else:
    print(f"'{project_root}' is already in sys.path.")

print("\n--- Current sys.path ---")
for p in sys.path:
    print(p)
print("------------------------\n")



import pandas as pd
from utils.recommendation_logic import OutfitRecommender # Import the class

# --- Configuration ---
PROCESSED_DATA_PATH = os.path.join('data', 'vastra_processed_data_with_local_paths.csv')

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Vastra: Step 4 - Demonstrating Outfit Recommendations ---")
    
    try:
        # Initialize the Outfit Recommender
        recommender = OutfitRecommender(PROCESSED_DATA_PATH)
        
        # Load the full DataFrame again to pick sample product IDs for demonstration
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH)
            df['product_id'] = df['product_id'].astype(str) # Ensure product_id is string
            
            df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce').fillna(-1).astype(int) 
            
        except FileNotFoundError:
            print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}. Cannot pick sample products.")
            exit()
        except Exception as e:
            print(f"Error loading processed data file or converting category_id: {e}")
            exit()

        sample_product_ids = []
        if not df.empty:
           
            dress_samples = df[df['category_id'] == 30] 
            # category_id 56 for Jeans (Bottomwear)
            bottomwear_samples = df[df['category_id'] == 56] 
            
            
            if not dress_samples.empty:
                sample_product_ids.append(dress_samples.sample(1)['product_id'].iloc[0])
            if not bottomwear_samples.empty:
                sample_product_ids.append(bottomwear_samples.sample(1)['product_id'].iloc[0])
           
            
            if not sample_product_ids: # Fallback if even 30 and 56 are not found (highly unlikely if data is there)
                 print("Warning: No products found for category_id 30 (Dresses) or 56 (Jeans). Picking a random sample if available.")
                 if not df.empty:
                     sample_product_ids.append(df.sample(1)['product_id'].iloc[0])
                     
        if not sample_product_ids:
            print("No valid products found in the dataset to demonstrate recommendations, even with random pick.")
            exit()

        for product_id_to_test in sample_product_ids:
            recommendations = recommender.get_outfit_recommendations(
                product_id_to_test, 
                num_recommendations_per_type=3 
            )

            if recommendations:
                print("\n--- Outfit Recommendations ---")
                for comp_type, items in recommendations.items():
                    print(f"\nRecommended {comp_type}:")
                    if items:
                        for item in items:
                            print(f"  - Name: {item['product_name']}")
                            print(f"    ID: {item['product_id']}")
                            print(f"    Image Path: {item['local_image_path']}")
                    else:
                       
                        print(f"  No {comp_type} recommendations found in your current dataset.")
                print("-" * 30)
            else:
                print(f"\nCould not generate recommendations for product ID: {product_id_to_test}")
                print("-" * 30)

    except ValueError as e:
        print(f"Error initializing OutfitRecommender: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")