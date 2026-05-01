import pandas as pd
import numpy as np
import os
import json

class OutfitRecommender:
    def __init__(self, processed_data_path):
        self.df = self._load_data(processed_data_path)
        if self.df is None:
            raise ValueError("Failed to load processed data for Outfit Recommender.")

        self.df['product_id'] = self.df['product_id'].astype(str)

        def parse_style_attributes(attr_string):
            if pd.isna(attr_string):
                return {}
            try:
                return json.loads(attr_string)
            except Exception:
                return {}

        self.df['parsed_style_attributes'] = self.df['style_attributes'].apply(parse_style_attributes)

        # ── Only map category IDs that ACTUALLY EXIST in your dataset ──────
        self.category_mapping = {
            30: 'Dresses',
            56: 'Bottomwear',
        }

        # ── Rules must only reference types present in category_mapping ─────
        self.category_rules = {
            'Dresses':    ['Bottomwear'],   # Dress  → recommend Jeans
            'Bottomwear': ['Dresses'],      # Jeans  → recommend Dresses
        }

        print("OutfitRecommender initialized.")

    def _load_data(self, path):
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} products for outfit recommendations.")
            return df
        except FileNotFoundError:
            print(f"Error: Processed data file not found at {path}.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _get_broad_category(self, product_category_id):
        if pd.isna(product_category_id):
            return None
        try:
            return self.category_mapping.get(int(product_category_id), None)
        except (ValueError, TypeError):
            return None

    def _get_product_style_keywords(self, product_row):
        styles = []
        parsed_attrs = product_row.get('parsed_style_attributes', {})
        for key in ('occasion', 'pattern', 'fit', 'trend', 'silhouette'):
            val = parsed_attrs.get(key)
            if val:
                styles.append(str(val).lower())
        return list(set(styles))

    def get_outfit_recommendations(self, query_product_id, num_recommendations_per_type=2):
        query_product = self.df[self.df['product_id'] == str(query_product_id)]

        if query_product.empty:
            print(f"Product ID '{query_product_id}' not found.")
            return None

        query_row      = query_product.iloc[0]
        query_cat_id   = query_row['category_id']
        broad_cat      = self._get_broad_category(query_cat_id)
        query_styles   = self._get_product_style_keywords(query_row)

        print(f"\nQuery: '{query_row['product_name']}' | cat_id={query_cat_id} | broad={broad_cat} | styles={query_styles}")

        if broad_cat not in self.category_rules:
            print(f"No rules defined for '{broad_cat}'. Available: {list(self.category_rules.keys())}")
            return {}

        complementary_types = self.category_rules[broad_cat]
        recommendations = {}

        for comp_type in complementary_types:
            # Find all products whose category maps to comp_type
            pool = self.df[
                self.df['category_id'].apply(
                    lambda x: self._get_broad_category(x) == comp_type if pd.notna(x) else False
                )
            ].copy()

            # Exclude the query product itself
            pool = pool[pool['product_id'] != str(query_product_id)]

            if pool.empty:
                recommendations[comp_type] = []
                continue

            # Prioritise style-compatible items if we have style info
            if query_styles:
                matches_style = pool.apply(
                    lambda row: any(s in self._get_product_style_keywords(row) for s in query_styles),
                    axis=1
                )
                pool = pd.concat([pool[matches_style], pool[~matches_style]]) \
                         .drop_duplicates(subset=['product_id'])

            n = min(num_recommendations_per_type, len(pool))
            sampled = pool.head(n)   # head() keeps priority order from concat above
            recommendations[comp_type] = sampled[
                ['product_id', 'product_name', 'local_image_path', 'pdp_url']
            ].to_dict(orient='records')

        return recommendations


# ── Convenience helper ────────────────────────────────────────────────────────
def add_custom_category_rule(recommender_instance, category_name, complementary_list, specific_mappings):
    recommender_instance.category_rules[category_name] = complementary_list
    recommender_instance.category_mapping.update(specific_mappings)
    print(f"Added custom rule for '{category_name}'.")


if __name__ == '__main__':
    PROCESSED_DATA_PATH = os.path.join('..', 'data', 'vastra_processed_data_with_local_paths.csv')
    try:
        rec = OutfitRecommender(PROCESSED_DATA_PATH)
        df  = rec.df

        for cat_id, label in [(30, 'Dress'), (56, 'Jeans')]:
            subset = df[df['category_id'] == cat_id]
            if subset.empty:
                print(f"No {label} products found (category_id={cat_id})")
                continue
            pid = subset.sample(1)['product_id'].iloc[0]
            print(f"\n--- {label} (ID: {pid}) ---")
            recs = rec.get_outfit_recommendations(pid, num_recommendations_per_type=2)
            for comp_type, items in (recs or {}).items():
                print(f"  {comp_type}:")
                for item in items:
                    print(f"    · {item['product_name']}")
    except Exception as e:
        print(f"Error: {e}")