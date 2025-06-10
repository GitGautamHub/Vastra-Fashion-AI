

import faiss
import numpy as np
import pandas as pd
import os

# --- Configuration ---
EMBEDDINGS_FILE = os.path.join('models', 'vastra_image_embeddings.npy')
PRODUCT_IDS_FILE = os.path.join('models', 'vastra_product_ids_for_embeddings.npy')
FAISS_INDEX_FILE = os.path.join('models', 'vastra_faiss_index.bin')
PROCESSED_DATA_PATH = os.path.join('data', 'vastra_processed_data_with_local_paths.csv')

# --- Main execution block ---
if __name__ == "__main__":
    print(f"Loading image embeddings from: {EMBEDDINGS_FILE}")
    try:
        # Load the embeddings and product IDs
        embeddings = np.load(EMBEDDINGS_FILE)
        product_ids = np.load(PRODUCT_IDS_FILE, allow_pickle=True) # allow_pickle=True might be needed for string arrays
        
        print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
        print(f"Loaded {len(product_ids)} corresponding product IDs.")
        
        if embeddings.shape[0] != len(product_ids):
            print("Error: Number of embeddings does not match number of product IDs. Check Step 2 output.")
            exit()
        
    except FileNotFoundError:
        print(f"Error: Embeddings or product IDs files not found. Please run scripts/step_2_feature_extraction.py first.")
        exit()
    except Exception as e:
        print(f"Error loading embeddings or product IDs: {e}")
        exit()

    # Ensure embeddings are float32, which Faiss expects
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
        print("Converted embeddings to float32.")

    # ---  Create and Train Faiss Index ---
    # Determine the dimensionality of the embeddings
    dimension = embeddings.shape[1] # This will be 2048 for ResNet50

 
    # For very large datasets, you'd consider IVFFlat, HNSW, etc., but they require more complex setup.
    index = faiss.IndexFlatL2(dimension) 
    print(f"\nCreated Faiss index of type IndexFlatL2 with dimension {dimension}.")

    # Add the embeddings to the index
    print("Adding embeddings to the Faiss index...")
    index.add(embeddings)
    print(f"Total vectors in the index: {index.ntotal}")

    # ---  Save the Faiss Index ---
    print(f"\nSaving Faiss index to: {FAISS_INDEX_FILE}")
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("Faiss index saved successfully!")

    # --- Optional: Test the index (Load and Query) ---
    print("\n--- Testing the Faiss index ---")
    print(f"Loading index from: {FAISS_INDEX_FILE}")
    try:
        loaded_index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"Index loaded. Total vectors: {loaded_index.ntotal}")
    except Exception as e:
        print(f"Error loading saved index: {e}")
        exit()

    # Perform a test query: Find the top 5 most similar items to the first item in your dataset
    # (assuming the first item's embedding is representative)
    query_embedding = embeddings[0:1] # Take the first embedding as a query (needs to be 2D array)
    
    print(f"Querying for top 5 similar items to product ID: {product_ids[0]}")
    
    # D: distances, I: indices of the nearest neighbors
    k = 5 # Number of nearest neighbors to retrieve
    D, I = loaded_index.search(query_embedding, k)

    print(f"\nDistances to top {k} similar items: {D[0]}")
    print(f"Indices of top {k} similar items: {I[0]}")

    # Map indices back to product IDs
    retrieved_product_ids = [product_ids[i] for i in I[0]]
    print(f"Product IDs of top {k} similar items: {retrieved_product_ids}")

    # You can also load the original processed data to get more details about these products
    try:
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        print("\nDetails of top 5 similar items:")
        df_processed['product_id'] = df_processed['product_id'].astype(str)
        
        # Using isin for efficient filtering
        similar_items_df = df_processed[df_processed['product_id'].isin(retrieved_product_ids)].copy()
        
    
        similar_items_df['product_id'] = pd.Categorical(
            similar_items_df['product_id'], 
            categories=retrieved_product_ids, 
            ordered=True
        )
        similar_items_df.sort_values('product_id', inplace=True)
        
        # Display relevant columns
        print(similar_items_df[['product_id', 'product_name', 'category_id', 'local_image_path']])
        
    except FileNotFoundError:
        print(f"Warning: {PROCESSED_DATA_PATH} not found. Cannot retrieve product details.")
    except Exception as e:
        print(f"Error retrieving product details: {e}")