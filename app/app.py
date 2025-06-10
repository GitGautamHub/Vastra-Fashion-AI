

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# --- Streamlit Page Configuration  ---
import streamlit as st 
st.set_page_config(layout="wide", page_title="Vastra: Intelligent Styling Assistant")


from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import faiss
from io import BytesIO

from utils.recommendation_logic import OutfitRecommender 

# --- Configuration ---
PROCESSED_DATA_PATH = os.path.join('data', 'vastra_processed_data_with_local_paths.csv')
EMBEDDINGS_FILE = os.path.join('models', 'vastra_image_embeddings.npy')
PRODUCT_IDS_FILE = os.path.join('models', 'vastra_product_ids_for_embeddings.npy')
FAISS_INDEX_FILE = os.path.join('models', 'vastra_faiss_index.bin')
IMAGES_DIR = 'downloaded_fashion_images' 

# --- Load Models and Data ---

@st.cache_resource 
def load_feature_extractor_model():
    """Loads the pre-trained ResNet50 model for feature extraction."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1])) 
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Feature Extractor Model loaded on {device}.") 
    return model, device

@st.cache_data 
def load_all_data_and_index(processed_data_path, embeddings_file, product_ids_file, faiss_index_file):
    """Loads all necessary data and the Faiss index."""
    try:
        df = pd.read_csv(processed_data_path)
        df['product_id'] = df['product_id'].astype(str)
        df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce').fillna(-1).astype(int)
        
        df['launch_on'] = pd.to_datetime(df['launch_on'], errors='coerce')


        print(f"Loaded {len(df)} products from processed data.") 

        embeddings = np.load(embeddings_file)
        product_ids = np.load(product_ids_file, allow_pickle=True)
        print(f"Loaded {len(embeddings)} embeddings.") 

        faiss_index = faiss.read_index(faiss_index_file)
        print(f"Loaded Faiss index with {faiss_index.ntotal} vectors.") 
        
        return df, embeddings, product_ids, faiss_index
    except FileNotFoundError as e:
        st.error(f"Error: Required data file not found. Please ensure all previous steps (1-3) were completed successfully and files are in correct locations. Missing file: {e.filename}")
        st.stop() 
    except Exception as e:
        st.error(f"Error loading core data or Faiss index: {e}")
        st.stop()

# Load everything once at the start of the app
feature_extractor_model, device = load_feature_extractor_model()
df_products, all_embeddings, all_product_ids, faiss_index = load_all_data_and_index(
    PROCESSED_DATA_PATH, EMBEDDINGS_FILE, PRODUCT_IDS_FILE, FAISS_INDEX_FILE
)

# Initialize OutfitRecommender
@st.cache_resource 
def load_recommender():
    try:
        return OutfitRecommender(PROCESSED_DATA_PATH)
    except ValueError as e:
        st.error(f"Error initializing Outfit Recommender: {e}. Please check category mappings in utils/recommendation_logic.py.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while initializing recommender: {e}")
        st.stop()

outfit_recommender = load_recommender()

# Define image transformations for the model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Helper Functions for App Logic ---

def extract_query_features(image_pil, model, preprocess, device):
    """Extracts features from a PIL Image."""
    try:
        img_tensor = preprocess(image_pil.convert('RGB'))
        img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)
        
        return features.squeeze().cpu().numpy().astype('float32')
    except Exception as e:
        st.error(f"Error extracting features from uploaded image: {e}")
        return None

def perform_visual_search(query_features, index, k=5):
    """Performs similarity search using Faiss index."""
    if query_features is None:
        return None, None

    # Faiss search expects a 2D array, even for a single query
    D, I = index.search(query_features.reshape(1, -1), k) # D: distances, I: indices
    return D, I

# --- NEW: Function to get trending items ---
@st.cache_data # Cache this result as it's static per run
def get_trending_items(df, num_items=6):
    """
    Identifies and returns the latest launched items as trending.
    Assumes 'launch_on' is a datetime column.
    """
    # Filter out rows with invalid launch_on dates (NaT)
    trending_df = df.dropna(subset=['launch_on']).copy()
    
    # Sort by launch_on in descending order (latest first)
    trending_df = trending_df.sort_values(by='launch_on', ascending=False)
    
    # Get unique products up to num_items
    # Use drop_duplicates to ensure unique products if there are exact duplicates by name/image
    trending_df = trending_df.drop_duplicates(subset=['product_id']).head(num_items)
    
    if trending_df.empty:
        return None
    return trending_df[['product_id', 'product_name', 'local_image_path', 'category_id', 'pdp_url']]

# Get trending items at app startup
trending_products_df = get_trending_items(df_products, num_items=6)


# --- Streamlit UI Layout ---

st.title("👗 Vastra: Your Intelligent Styling Assistant")
st.markdown("Upload any fashion item image, and Vastra will find similar products and suggest complementary outfits!")

# --- Initialize session state for search history ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- Image Uploader ---
uploaded_file = st.file_uploader("Upload a fashion item image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        uploaded_image = Image.open(BytesIO(uploaded_file.read()))
        st.subheader("Your Uploaded Item:")
        st.image(uploaded_image, caption=f"Uploaded Image ({uploaded_file.name})", use_container_width=True)

        # --- Feature Extraction and Search ---
        st.subheader("Searching for Similar Items...")
        with st.spinner("Extracting features and searching..."):
            query_features = extract_query_features(uploaded_image, feature_extractor_model, preprocess, device)

            if query_features is not None:
                D, I = perform_visual_search(query_features, faiss_index, k=6) 

                if I is not None:
                    st.success("Search Complete!")
                    st.subheader("Visually Similar Products:")

                    similar_product_indices = I[0]
                    
                    unique_similar_product_ids = []
                    top_similar_product_id_for_history = None 
                    for idx in similar_product_indices:
                        prod_id = all_product_ids[idx]
                        if top_similar_product_id_for_history is None: 
                            top_similar_product_id_for_history = prod_id

                        if prod_id not in unique_similar_product_ids and (idx != similar_product_indices[0] or D[0][0] > 0.001):
                             unique_similar_product_ids.append(prod_id)
                        if len(unique_similar_product_ids) >= 5: 
                            break

                    display_products_df = df_products[df_products['product_id'].isin(unique_similar_product_ids)].copy()

                    if not display_products_df.empty:
                        cols = st.columns(min(len(display_products_df), 5)) 
                        for i, (_, row) in enumerate(display_products_df.head(5).iterrows()):
                            with cols[i]:
                                try:
                                    img_path = row['local_image_path']
                                    if os.path.exists(img_path):
                                        st.image(Image.open(img_path).convert('RGB'), caption=f"{row['product_name']}", width=150)
                                        st.markdown(f"**Category:** {row['category_id']}")
                                        if pd.notna(row['pdp_url']):
                                            st.markdown(f"[View Product]({row['pdp_url']})")
                                        else:
                                            st.markdown("No product link available.")
                                    else:
                                        st.warning(f"Image not found for {row['product_name']}")
                                except Exception as e:
                                    st.error(f"Could not display image for {row['product_name']}: {e}")
                    else:
                        st.info("No distinct similar products found in the database.")
                else:
                    st.warning("Could not perform visual search.")

                # --- Update Search History ---
                if top_similar_product_id_for_history:
                    if top_similar_product_id_for_history not in st.session_state.search_history: 
                        st.session_state.search_history.append(top_similar_product_id_for_history)
                    if len(st.session_state.search_history) > 5:
                        st.session_state.search_history = st.session_state.search_history[-5:] 

                # --- Outfit Recommendations ---
                st.subheader("Outfit Recommendations for Your Item:")
                
                top_similar_product_id = all_product_ids[I[0][0]] 
                
                outfit_recs = outfit_recommender.get_outfit_recommendations(
                    top_similar_product_id, num_recommendations_per_type=2 
                )

                if outfit_recs:
                    filtered_outfit_recs = {k: v for k, v in outfit_recs.items() if v}
                    
                    if filtered_outfit_recs:
                        rec_cols = st.columns(len(filtered_outfit_recs.keys())) 
                        col_idx = 0
                        for comp_type, items in filtered_outfit_recs.items():
                            with rec_cols[col_idx]:
                                st.markdown(f"**{comp_type}:**")
                                if items:
                                    for item in items:
                                        try:
                                            img_path = item['local_image_path']
                                            if os.path.exists(img_path):
                                                st.image(Image.open(img_path).convert('RGB'), caption=f"{item['product_name']}", width=120)
                                                if pd.notna(item['pdp_url']):
                                                    st.markdown(f"[View Product]({item['pdp_url']})")
                                                else:
                                                    st.markdown("No product link available.")
                                            else:
                                                st.warning(f"Image not found for {item['product_name']}")
                                        except Exception as e:
                                            st.error(f"Could not display image for {item['product_name']}: {e}")
                                else:
                                    pass 
                            col_idx += 1
                    else:
                        st.info("No outfit recommendations found for this item's category in the dataset.")
                else:
                    st.info("Could not generate recommendations for this item's category.")

                # --- Personalized Recommendations (EXISTING SECTION) ---
                st.subheader("Recommendations based on your recent searches:")

                if len(st.session_state.search_history) > 1:
                    unique_past_search_ids = list(st.session_state.search_history[:-1])
                    
                    if unique_past_search_ids:
                        all_personalized_recs_df = pd.DataFrame()
                        
                        past_searched_products_df = df_products[df_products['product_id'].isin(unique_past_search_ids)].copy()
                        
                        if not past_searched_products_df.empty:
                            past_categories = past_searched_products_df['category_id'].unique()
                            past_brands = past_searched_products_df['brand'].unique()
                            
                            excluded_ids = st.session_state.search_history + unique_similar_product_ids
                            
                            personalized_suggestions = df_products[
                                (df_products['category_id'].isin(past_categories)) | 
                                (df_products['brand'].isin(past_brands))
                            ].copy()
                            
                            personalized_suggestions = personalized_suggestions[~personalized_suggestions['product_id'].isin(excluded_ids)]
                            
                            if not personalized_suggestions.empty:
                                num_to_sample = min(6, len(personalized_suggestions))
                                personalized_display_df = personalized_suggestions.sample(num_to_sample, random_state=43)
                                
                                st.markdown("Here are some items similar to what you've searched before:")
                                p_cols = st.columns(len(personalized_display_df) or 1)
                                for i, (_, row) in enumerate(personalized_display_df.iterrows()):
                                    with p_cols[i]:
                                        try:
                                            img_path = row['local_image_path']
                                            if os.path.exists(img_path):
                                                st.image(Image.open(img_path).convert('RGB'), caption=f"{row['product_name']}", width=120)
                                                st.markdown(f"**Category:** {row['category_id']}")
                                                if pd.notna(row['pdp_url']):
                                                    st.markdown(f"[View Product]({row['pdp_url']})")
                                                else:
                                                    st.markdown("No product link available.")
                                            else:
                                                st.warning(f"Image not found for {row['product_name']}")
                                        except Exception as e:
                                            st.error(f"Could not display image for {row['product_name']}: {e}")
                            else:
                                st.info("No new personalized recommendations found based on your search history.")
                        else:
                            st.info("Your past searches contain items not found in the database for personalized recommendations.")
                    else:
                         st.info("Continue searching to get personalized recommendations!")
                else:
                    st.info("Perform a few more searches to see personalized recommendations based on your history.")


            else:
                st.error("Failed to extract features from the uploaded image.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e) 
else:
    st.info("Please upload an image to get started!")

# --- NEW: Trend Awareness Section ---

if trending_products_df is not None and not trending_products_df.empty:
    st.subheader("✨ New Arrivals (Trending Items):")
    t_cols = st.columns(min(len(trending_products_df), 6))
    for i, (_, row) in enumerate(trending_products_df.iterrows()):
        with t_cols[i]:
            try:
                img_path = row['local_image_path']
                if os.path.exists(img_path):
                    st.image(Image.open(img_path).convert('RGB'), caption=f"{row['product_name']}", width=120)
                    st.markdown(f"**Category:** {row['category_id']}")
                    if pd.notna(row['pdp_url']):
                        st.markdown(f"[View Product]({row['pdp_url']})")
                    else:
                        st.markdown("No product link available.")
                else:
                    st.warning(f"Image not found for {row['product_name']}")
            except Exception as e:
                st.error(f"Could not display image for {row['product_name']}: {e}")
else:
    st.info("No new arrivals/trending items found in the dataset.")


st.markdown("""
<style>
.reportview-container .main .block-container {
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 1rem;
}
.stImage {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 5px;
}
.stImage img {
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)