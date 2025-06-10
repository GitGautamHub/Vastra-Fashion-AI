# utils/recommendation_logic.py

import pandas as pd
import numpy as np
import os
import json # NEW: Import json to parse style_attributes

class OutfitRecommender:
    """
    A class to provide basic outfit recommendations based on product categories.
    Now also incorporates style compatibility using 'style_attributes'.
    """
    def __init__(self, processed_data_path):
        """
        Initializes the recommender by loading the processed product data.
        """
        self.df = self._load_data(processed_data_path)
        if self.df is None:
            raise ValueError("Failed to load processed data for Outfit Recommender.")
        
        # Ensure product_id is string type for consistent matching
        self.df['product_id'] = self.df['product_id'].astype(str)
        
        # --- NEW: Process style_attributes column ---
        # Convert string representation of dictionary to actual dictionary
        # Handle NaN values and invalid JSON strings
        def parse_style_attributes(attr_string):
            if pd.isna(attr_string):
                return {}
            try:
                # ast.literal_eval is safer than json.loads for arbitrary strings,
                # but json.loads is standard for JSON. Given the data source, JSON is expected.
                # If you encounter issues, try import ast and use ast.literal_eval(attr_string)
                return json.loads(attr_string)
            except json.JSONDecodeError:
                # print(f"Warning: Could not parse style_attributes: {attr_string}") # For debugging
                return {} # Return empty dict for unparseable strings
            except Exception as e:
                # print(f"Unexpected error parsing style_attributes: {e}") # For debugging
                return {}
        
        self.df['parsed_style_attributes'] = self.df['style_attributes'].apply(parse_style_attributes)
        # --- END NEW ---

        # Define basic category relationships for outfit recommendations
        self.category_rules = {
            'Dresses': ['Footwear', 'Accessories'], 
            'Bottomwear': ['Topwear', 'Footwear', 'Accessories'], 
            # You can add more general categories here if you plan to add more data later.
        }
        
        # Mapping for actual numerical category_id values to broader types used in rules.
        self.category_mapping = {
            30: 'Dresses',   # Category ID 30 is for Dresses
            56: 'Bottomwear' # Category ID 56 is for Jeans (which fall under Bottomwear)
            # Add more mappings as you add more data
        }
        
        print("OutfitRecommender initialized.")

    def _load_data(self, path):
        """Loads the product data from the specified path."""
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} products for outfit recommendations.")
            return df
        except FileNotFoundError:
            print(f"Error: Processed data file not found at {path}.")
            return None
        except Exception as e:
            print(f"Error loading data for outfit recommendations: {e}")
            return None

    def _get_broad_category(self, product_category_id):
        """
        Maps a specific numerical product's category_id to a broader defined category for rules.
        """
        if pd.isna(product_category_id):
            return None

        try:
            product_category_id_int = int(product_category_id)
        except ValueError:
            return None 
        
        return self.category_mapping.get(product_category_id_int, None)

    # --- NEW: Helper function to get style attributes from a product ---
    def _get_product_style_keywords(self, product_row):
        """
        Extracts relevant style keywords from a product row's parsed_style_attributes.
        Customize this based on what style attributes are important.
        """
        styles = []
        parsed_attrs = product_row['parsed_style_attributes']
        
        # Example: Extracting common style attributes
        # You might need to inspect your style_attributes to find relevant keys
        if 'occasion' in parsed_attrs and parsed_attrs['occasion']:
            styles.append(parsed_attrs['occasion'].lower())
        if 'pattern' in parsed_attrs and parsed_attrs['pattern']:
            styles.append(parsed_attrs['pattern'].lower())
        if 'fit' in parsed_attrs and parsed_attrs['fit']:
            styles.append(parsed_attrs['fit'].lower())
        if 'trend' in parsed_attrs and parsed_attrs['trend']: # Assuming 'trend' is a key
            styles.append(parsed_attrs['trend'].lower())
        if 'silhouette' in parsed_attrs and parsed_attrs['silhouette']:
             styles.append(parsed_attrs['silhouette'].lower())
        # Add more relevant keys from your style_attributes as needed
        
        return list(set(styles)) # Return unique style keywords

    def get_outfit_recommendations(self, query_product_id, num_recommendations_per_type=2):
        """
        Generates outfit recommendations for a given product ID,
        now incorporating style compatibility.
        """
        query_product = self.df[self.df['product_id'] == query_product_id]

        if query_product.empty:
            print(f"Query product ID '{query_product_id}' not found in the dataset.")
            return None

        query_product_category_id = query_product['category_id'].iloc[0] 
        query_product_broad_category = self._get_broad_category(query_product_category_id)
        
        # --- NEW: Get query product's style attributes ---
        query_product_styles = self._get_product_style_keywords(query_product.iloc[0]) # Pass the first row
        print(f"Query Product Styles: {query_product_styles}")
        # --- END NEW ---

        print(f"\nQuery Product: '{query_product['product_name'].iloc[0]}' (ID: {query_product_id}, Category ID: {query_product_category_id}, Broad Category: {query_product_broad_category})")
        
        if query_product_broad_category not in self.category_rules:
            print(f"No outfit rules defined for broad category: {query_product_broad_category}")
            return {}

        complementary_types = self.category_rules[query_product_broad_category]
        recommendations = {}

        for comp_type in complementary_types:
            potential_recs_df = self.df[
                self.df['category_id'].apply(lambda x: self._get_broad_category(x) == comp_type if pd.notna(x) else False)
            ].copy()
            
            potential_recs_df = potential_recs_df[potential_recs_df['product_id'] != query_product_id]
            
            # --- NEW: Filter recommendations by style compatibility ---
            if query_product_styles and not potential_recs_df.empty:
                # Find recommendations that share at least one style keyword
                
                # Create a list of booleans indicating if a product matches any query style
                style_matches = potential_recs_df.apply(
                    lambda row: any(style in self._get_product_style_keywords(row) for style in query_product_styles), 
                    axis=1
                )
                
                # Prioritize style-matching items
                style_matching_recs = potential_recs_df[style_matches]
                non_style_matching_recs = potential_recs_df[~style_matches]
                
                # Concatenate, prioritizing style matches
                potential_recs_df = pd.concat([style_matching_recs, non_style_matching_recs])
                potential_recs_df = potential_recs_df.drop_duplicates(subset=['product_id']) # Ensure uniqueness after concat
            # --- END NEW ---


            if not potential_recs_df.empty:
                num_to_sample = min(num_recommendations_per_type, len(potential_recs_df))
                if num_to_sample > 0:
                    # Random state for reproducibility, but now applies to the prioritized list
                    sampled_recs = potential_recs_df.sample(n=num_to_sample, random_state=42) 
                    recommendations[comp_type] = sampled_recs[['product_id', 'product_name', 'local_image_path', 'pdp_url']].to_dict(orient='records')
                else:
                    recommendations[comp_type] = []
            else:
                recommendations[comp_type] = [] 

        return recommendations

# Example of how to add a specific category to rules and mapping (if you expand your data)
def add_custom_category_rule(recommender_instance, category_name, complementary_list, specific_mappings):
    recommender_instance.category_rules[category_name] = complementary_list
    recommender_instance.category_mapping.update(specific_mappings)
    print(f"Added custom rule for '{category_name}'.")

if __name__ == '__main__':
    # This block is for testing this module directly
    PROCESSED_DATA_PATH = os.path.join('..', 'data', 'vastra_processed_data_with_local_paths.csv')
    
    try:
        recommender = OutfitRecommender(PROCESSED_DATA_PATH)

        sample_product_ids_to_test = []
        df_full = recommender.df 

        dress_sample = df_full[df_full['category_id'] == 30].sample(1)['product_id'].iloc[0] if not df_full[df_full['category_id'] == 30].empty else None
        if dress_sample:
            sample_product_ids_to_test.append(dress_sample)

        jeans_sample = df_full[df_full['category_id'] == 56].sample(1)['product_id'].iloc[0] if not df_full[df_full['category_id'] == 56].empty else None
        if jeans_sample:
            sample_product_ids_to_test.append(jeans_sample)

        if not sample_product_ids_to_test:
            print("No valid Dress (category_id 30) or Jeans (category_id 56) products found in the dataset to test.")
            exit()

        for product_id_to_test in sample_product_ids_to_test:
            print(f"\n--- Testing Outfit Recommender for Product ID: {product_id_to_test} ---")
            recommendations = recommender.get_outfit_recommendations(product_id_to_test, num_recommendations_per_type=2)
            
            if recommendations:
                for comp_type, items in recommendations.items():
                    print(f"\nRecommended {comp_type}:")
                    if items:
                        for item in items:
                            print(f"  - Product: {item['product_name']} (ID: {item['product_id']})")
                            print(f"    Product URL: {item['pdp_url']}") 
                    else:
                        print(f"  No {comp_type} recommendations found in your current dataset.")
            else:
                print("No recommendations could be generated for this product.")
            print("-" * 50) 

    except ValueError as e:
        print(f"Initialization error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")