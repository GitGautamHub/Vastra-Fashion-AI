import os
import sys
import base64
import streamlit.components.v1 as components

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st

    
st.set_page_config(
    layout="wide",
    page_title="Vastra — Intelligent Styling",
    page_icon="👗",
    initial_sidebar_state="expanded"
)

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import faiss
from io import BytesIO

# ─── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DATA_PATH = os.path.join('data', 'vastra_processed_data_with_local_paths.csv')
EMBEDDINGS_FILE     = os.path.join('models', 'vastra_image_embeddings.npy')
PRODUCT_IDS_FILE    = os.path.join('models', 'vastra_product_ids_for_embeddings.npy')
FAISS_INDEX_FILE    = os.path.join('models', 'vastra_faiss_index.bin')
IMAGES_DIR          = 'downloaded_fashion_images'

# ─── Category constants ────────────────────────────────────────────────────────
CAT_DRESSES = 30
CAT_JEANS   = 56

# ─── Session state defaults ────────────────────────────────────────────────────
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = {}   # {product_id: {'name': ..., 'pdp_url': ...}}
if 'style_seed' not in st.session_state:
    st.session_state.style_seed = 42


# ─── Theme variables ───────────────────────────────────────────────────────────
DARK_THEME = {
    "--ink":         "#0e0d0b",
    "--parchment":   "#f5f0e8",
    "--gold":        "#c9a84c",
    "--gold-lt":     "#e8d5a3",
    "--ember":       "#8b2e12",
    "--mist":        "#d9d3c7",
    "--card-bg":     "#1a1814",
    "--card-border": "rgba(201,168,76,0.18)",
    "--input-bg":    "#141210",
    "--sidebar-bg":  "#111009",
}
LIGHT_THEME = {
    "--ink":         "#faf7f2",
    "--parchment":   "#1a1208",
    "--gold":        "#9a6f1a",
    "--gold-lt":     "#7a530e",
    "--ember":       "#8b2e12",
    "--mist":        "#5a4e3a",
    "--card-bg":     "#ffffff",
    "--card-border": "rgba(154,111,26,0.2)",
    "--input-bg":    "#f0ebe0",
    "--sidebar-bg":  "#ede8de",
}

theme = DARK_THEME if st.session_state.dark_mode else LIGHT_THEME
css_vars = "\n".join([f"    {k}: {v};" for k, v in theme.items()])

# ─── Inject Global CSS ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {{
{css_vars}
    --radius: 12px;
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--ink) !important;
    color: var(--parchment) !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 2.5rem 3rem !important; max-width: 1400px !important; }}

/* ── Hero ── */
.vastra-hero {{
    background: {'linear-gradient(135deg, #0e0d0b 40%, #1f1a12 100%)' if st.session_state.dark_mode else 'linear-gradient(135deg, #faf7f2 40%, #ede8de 100%)'};
    border-bottom: 1px solid var(--card-border);
    padding: 3rem 0 2rem;
    margin-bottom: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.vastra-hero::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(201,168,76,0.08) 0%, transparent 70%);
    pointer-events: none;
}}
.vastra-logo {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 5rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.28em !important;
    color: var(--gold) !important;
    line-height: 1 !important;
    margin-bottom: 0.2rem;
}}
.vastra-tagline {{
    font-size: 0.78rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--mist);
    opacity: 0.75;
}}
.vastra-divider {{
    display: block; width: 60px; height: 1px;
    background: var(--gold);
    margin: 1rem auto 0; opacity: 0.55;
}}

/* ── Section headings ── */
.section-heading {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.85rem !important;
    font-weight: 400 !important;
    color: var(--gold-lt) !important;
    letter-spacing: 0.04em;
    margin: 2.8rem 0 0.2rem;
    display: flex; align-items: center; gap: 0.65rem;
}}
.section-heading::after {{
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--card-border), transparent);
    margin-left: 0.6rem;
}}
.section-sub {{
    font-size: 0.75rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--mist);
    opacity: 0.55; margin-bottom: 1.5rem;
}}

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {{
    background: var(--card-bg) !important;
    border: 1.5px dashed var(--card-border) !important;
    border-radius: var(--radius) !important;
    padding: 2.5rem !important;
    transition: border-color 0.3s;
}}
[data-testid="stFileUploadDropzone"]:hover {{ border-color: var(--gold) !important; }}
[data-testid="stFileUploadDropzone"] * {{ color: var(--mist) !important; }}

/* ── Text input styling ── */
[data-testid="stTextInput"] input {{
    background: var(--input-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
    color: var(--parchment) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.3s;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.12) !important;
}}
[data-testid="stTextInput"] input::placeholder {{ color: var(--mist); opacity: 0.5; }}

/* ── Score bar ── */
.score-bar-wrap {{
    display: inline-flex; align-items: center; gap: 0.8rem;
    padding: 0.6rem 1.2rem;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px; margin-bottom: 1rem;
}}
.score-number {{
    font-size: 1.4rem; font-weight: 600; color: var(--gold);
    font-family: 'Cormorant Garamond', serif;
}}
.score-label {{
    font-size: 0.7rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--mist);
}}
.score-track {{
    height: 4px; width: 120px; background: #2a2620;
    border-radius: 2px; margin-top: 4px; overflow: hidden;
}}
.score-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--ember), var(--gold));
    border-radius: 2px;
    transition: width 0.6s ease;
}}

/* ── Share box ── */
.share-box {{
    margin: 1.5rem 0; padding: 1rem 1.5rem;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.8rem;
}}
.share-code {{
    font-family: monospace; font-size: 0.72rem;
    background: var(--input-bg); padding: 0.4rem 0.8rem;
    border-radius: 4px; color: var(--gold);
    border: 1px solid var(--card-border);
    word-break: break-all;
}}

/* ── Upload frame ── */
.upload-frame {{
    border: 1px solid var(--card-border);
    border-radius: var(--radius); overflow: hidden;
    max-width: 340px; margin: 0 auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--card-border) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--parchment) !important; }}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div {{
    background: var(--input-bg) !important;
    border-color: var(--card-border) !important;
    border-radius: var(--radius) !important;
}}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {{
    background: var(--gold) !important;
}}

/* ── Buttons ── */
.stButton>button {{
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.12em !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: background 0.2s, color 0.2s !important;
}}
.stButton>button:hover {{
    background: var(--gold) !important;
    color: var(--ink) !important;
}}

/* ── Info / alert ── */
.stAlert {{
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
    color: var(--parchment) !important;
}}

/* ── Trending badge ── */
.trending-badge {{
    display: inline-block; background: var(--ember); color: #fff;
    font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 20px;
    vertical-align: middle; margin-left: 0.5rem;
}}

[data-testid="column"] {{ padding: 0 0.4rem !important; }}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: var(--ink); }}
::-webkit-scrollbar-thumb {{ background: var(--card-border); border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)

# ─── Hero + Theme Toggle ────────────────────────────────────────────────────────
hero_col, toggle_col = st.columns([10, 1])

        
with hero_col:
    st.markdown("""
    <div class="vastra-hero">
        <div class="vastra-logo">VASTRA</div>
        <div class="vastra-tagline">Intelligent Styling · Visual Discovery · Outfit Curation</div>
        <span class="vastra-divider"></span>
    </div>
    """, unsafe_allow_html=True)

with toggle_col:
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    toggle_label = "🌙 Dark" if st.session_state.dark_mode else "☀️ Light"
    if st.button(toggle_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ─── Product card HTML builder ─────────────────────────────────────────────────
def product_card_html(img_path, name, category, pdp_url, match_score=None):
    try:
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext  = img_path.rsplit(".", 1)[-1].lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
            img_tag = f'<img src="data:{mime};base64,{b64}" alt="{name}">'
        else:
            img_tag = '<div style="height:200px;background:#2a2620;display:flex;align-items:center;justify-content:center;color:#5a5040;font-size:0.7rem;letter-spacing:.1em;">NO IMAGE</div>'
    except Exception:
        img_tag = '<div style="height:200px;background:#2a2620;"></div>'

    link_html = (
        f'<div class="product-card-link"><a href="{pdp_url}" target="_blank">Shop Now →</a></div>'
        if pdp_url and str(pdp_url) != "nan" else
        '<div class="product-card-link" style="color:#4a4438;font-size:.7rem;">Unavailable</div>'
    )
    badge_html = f'<div class="match-badge">{match_score}% match</div>' if match_score else ''
    CAT_LABELS = {30: "Dresses", 56: "Jeans"}
    cat_label  = CAT_LABELS.get(int(category), f"Category {category}") if category != "" else ""

    return f"""
    <div class="product-card">
        {badge_html}
        {img_tag}
        <div class="product-card-body">
            <div class="product-card-name" title="{name}">{name}</div>
            <div class="product-card-cat">{cat_label}</div>
            {link_html}
        </div>
    </div>
    """

# ─── render_card: uses components.html to guarantee rendering inside columns ───
def render_card(img_path, name, category, pdp_url, match_score=None, key=None):
    """Renders a product card via components.html so CSS is self-contained
    and immune to Streamlit's unsafe_allow_html column-nesting issues."""
    card_html = product_card_html(img_path, name, category, pdp_url, match_score)
    full_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&family=DM+Sans:wght@300;400&display=swap');
    :root {{
        --ink: #0e0d0b; --parchment: #f5f0e8; --gold: #c9a84c;
        --gold-lt: #e8d5a3; --ember: #8b2e12; --mist: #d9d3c7;
        --card-bg: #1a1814; --card-border: rgba(201,168,76,0.18); --radius: 12px;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: transparent; }}
    .product-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: var(--radius);
        overflow: hidden;
        position: relative;

        transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
    }}
    .product-card:hover {{
        transform: translateY(-6px) scale(1.04);
        border-color: var(--gold);
        box-shadow: 0 16px 50px rgba(0,0,0,.65);
    }}
    .product-card img {{
        width: 100%;
        aspect-ratio: 3/4;
        object-fit: cover;
        display: block;

        transition: transform .4s ease;
    }}
    .product-card-body {{ padding: .9rem 1rem 1rem; }}
    .product-card-name {{
        font-family: 'Cormorant Garamond', serif; font-size: .95rem;
        color: var(--parchment); margin-bottom: .25rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .product-card-cat {{
        font-size: .67rem; letter-spacing: .18em; text-transform: uppercase;
        color: var(--gold); opacity: .75; margin-bottom: .6rem;
    }}
    .product-card-link a {{
        font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
        color: var(--gold-lt); text-decoration: none;
        border-bottom: 1px solid rgba(201,168,76,.3); padding-bottom: 1px;
    }}
    .product-card-link a:hover {{ border-color: var(--gold); }}
    .match-badge {{
        position: absolute; top: 8px; right: 8px;
        background: rgba(14,13,11,.82); border: 1px solid var(--gold);
        color: var(--gold); font-size: .65rem; letter-spacing: .1em;
        padding: 3px 8px; border-radius: 20px; backdrop-filter: blur(4px);
    }}
    </style>
    {card_html}
    """
    components.html(full_html, height=380)


# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_feature_extractor_model():
    model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model  = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

@st.cache_data
def load_all_data_and_index(processed_data_path, embeddings_file, product_ids_file, faiss_index_file):
    try:
        df = pd.read_csv(processed_data_path)
        df['product_id']  = df['product_id'].astype(str)
        df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce').fillna(-1).astype(int)
        df['launch_on']   = pd.to_datetime(df['launch_on'], errors='coerce')
        embeddings  = np.load(embeddings_file)
        product_ids = np.load(product_ids_file, allow_pickle=True)
        faiss_index = faiss.read_index(faiss_index_file)
        return df, embeddings, product_ids, faiss_index
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}. Complete preprocessing steps first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

feature_extractor_model, device = load_feature_extractor_model()
df_products, all_embeddings, all_product_ids, faiss_index = load_all_data_and_index(
    PROCESSED_DATA_PATH, EMBEDDINGS_FILE, PRODUCT_IDS_FILE, FAISS_INDEX_FILE
)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Core helpers ──────────────────────────────────────────────────────────────
def extract_query_features(image_pil, model, transform, dev):
    try:
        img_tensor = transform(image_pil.convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            features = model(img_tensor)
        return features.squeeze().cpu().numpy().astype('float32')
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def perform_visual_search(query_features, index, k=6):
    if query_features is None:
        return None, None
    D, I = index.search(query_features.reshape(1, -1), k)
    return D, I

def calculate_outfit_score(base_product_id, complement_ids, df):
    base = df[df['product_id'] == str(base_product_id)]
    if base.empty or not complement_ids:
        return 60
    base_brand = base.iloc[0].get('brand', '')
    score = 60
    complements = df[df['product_id'].isin([str(x) for x in complement_ids])]
    if not complements.empty:
        cats_covered  = complements['category_id'].nunique()
        score        += min(cats_covered * 8, 24)
        brand_matches = (complements['brand'] == base_brand).sum()
        score        += int(brand_matches) * 5
    return min(int(score), 99)

def apply_filters(df, selected_cats, selected_brands, price_range):
    filtered = df.copy()
    if selected_cats:
        filtered = filtered[filtered['category_id'].isin(selected_cats)]
    if selected_brands:
        filtered = filtered[filtered['brand'].isin(selected_brands)]
    if price_range and 'price' in df.columns:
        filtered = filtered[
            (filtered['price'] >= price_range[0]) &
            (filtered['price'] <= price_range[1])
        ]
    return filtered

@st.cache_data
def get_trending_items(df, num_items=6):
    trending_df = df.dropna(subset=['launch_on']).copy()
    trending_df = trending_df.sort_values('launch_on', ascending=False)
    trending_df = trending_df.drop_duplicates(subset=['product_id']).head(num_items)
    if trending_df.empty:
        return None
    return trending_df[['product_id', 'product_name', 'local_image_path', 'category_id', 'pdp_url']]

trending_products_df = get_trending_items(df_products, num_items=6)


# ─── Sidebar: Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Cormorant Garamond',serif;font-size:1.6rem;
                color:var(--gold);letter-spacing:.15em;padding:.8rem 0 .4rem;">
        ✦ Refine
    </div>
    <div style="height:1px;background:var(--card-border);margin-bottom:1.2rem;"></div>
    """, unsafe_allow_html=True)

    all_brands = sorted(df_products['brand'].dropna().unique().tolist()) if 'brand' in df_products.columns else []

    st.markdown('<div style="font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--mist);margin-bottom:.5rem;">Category</div>', unsafe_allow_html=True)
    show_dresses = st.checkbox("👗 Dresses", value=True, key="cb_dresses")
    show_jeans   = st.checkbox("👖 Jeans",   value=True, key="cb_jeans")
    selected_cats = []
    if show_dresses: selected_cats.append(CAT_DRESSES)
    if show_jeans:   selected_cats.append(CAT_JEANS)

    if all_brands:
        st.markdown('<div style="font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--mist);margin:.8rem 0 .3rem;">Brand</div>', unsafe_allow_html=True)
        selected_brands = st.multiselect("Brand", all_brands, default=all_brands, label_visibility="collapsed")
    else:
        selected_brands = []

    price_range = None
    if 'price' in df_products.columns:
        min_p = int(df_products['price'].min())
        max_p = int(df_products['price'].max())
        st.markdown('<div style="font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--mist);margin:.8rem 0 .3rem;">Price Range (₹)</div>', unsafe_allow_html=True)
        price_range = st.slider("Price", min_p, max_p, (min_p, max_p), label_visibility="collapsed")

    st.markdown("<div style='height:1px;background:var(--card-border);margin:1.4rem 0;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Cormorant Garamond',serif;font-size:1.2rem;
                color:var(--gold-lt);letter-spacing:.1em;margin-bottom:.6rem;">
        🔖 Wishlist
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.wishlist:
        for pid, meta in st.session_state.wishlist.items():
            pdp = meta.get('pdp_url', '')
            name_short = meta.get('name', pid)[:28]
            if pdp and str(pdp) != 'nan':
                st.markdown(
                    f'<div style="padding:.45rem 0;border-bottom:1px solid var(--card-border);">'
                    f'<a href="{pdp}" target="_blank" style="font-size:.75rem;color:var(--gold-lt);'
                    f'text-decoration:none;letter-spacing:.04em;">'
                    f'🔖 {name_short}…</a></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="padding:.45rem 0;border-bottom:1px solid var(--card-border);'
                    f'font-size:.75rem;color:var(--mist);">🔖 {name_short}…</div>',
                    unsafe_allow_html=True
                )
        if st.button("🗑 Clear Wishlist"):
            st.session_state.wishlist = {}
            st.rerun()
    else:
        st.markdown('<div style="font-size:.72rem;color:var(--mist);opacity:.5;">No saved items yet.</div>', unsafe_allow_html=True)


def wishlist_button(pid, name, pdp_url, key_suffix):
    already_saved = pid in st.session_state.wishlist
    btn_label = "✓ Saved" if already_saved else "🔖 Save"

    if st.button(btn_label, key=f"wl_{key_suffix}_{pid}", use_container_width=True):
        if pid not in st.session_state.wishlist:
            st.session_state.wishlist[pid] = {
                'name': name,
                'pdp_url': str(pdp_url)
            }
            st.toast("Added to wishlist!", icon="🔖")
            st.rerun()


# ─── Shared Outfit via query params ────────────────────────────────────────────
query_params = st.query_params
if "outfit" in query_params:
    try:
        decoded    = base64.urlsafe_b64decode(query_params["outfit"] + "==").decode()
        shared_ids = decoded.split(",")
        shared_df  = df_products[df_products['product_id'].isin(shared_ids)]
        if not shared_df.empty:
            st.markdown('<div class="section-heading">🔗 Shared Outfit</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Someone shared this look with you</div>', unsafe_allow_html=True)
            s_cols = st.columns(min(len(shared_df), 5))
            for i, (_, row) in enumerate(shared_df.head(5).iterrows()):
                with s_cols[i]:
                    # ✅ FIX: render_card instead of st.markdown
                    render_card(
                        row['local_image_path'], row['product_name'],
                        row['category_id'], row.get('pdp_url', None)
                    )
                    wishlist_button(
                        row['product_id'],
                        row['product_name'],
                        row.get('pdp_url', None),
                        key_suffix=f"shared_{i}"
                    )
            st.markdown("<hr style='border-color:var(--card-border);margin:2rem 0;'>", unsafe_allow_html=True)
    except Exception:
        pass




# ─── Text Search ───────────────────────────────────────────────────────────────

# Build a dynamic placeholder from actual product names in dataset
@st.cache_data
def get_search_examples(df, n=4):
    """Pick n real product names as placeholder examples."""
    samples = (
        df['product_name']
        .dropna()
        .drop_duplicates()
        .sample(min(n, len(df)), random_state=1)
        .tolist()
    )
    # Shorten to first 3 words each
    short = [" ".join(s.split()[:3]) for s in samples]
    return short

def robust_search(df, query, max_results=6):
    """
    Multi-token search across product_name + brand + category label.
    Each token must match at least one column (AND logic across tokens).
    Partial substring match — handles multi-word queries gracefully.
    """
    CAT_NAME_MAP = {30: "dresses dress", 56: "jeans denim bottomwear"}

    # Precompute a single search corpus column once
    corpus_parts = [df['product_name'].fillna('')]
    if 'brand' in df.columns:
        corpus_parts.append(df['brand'].fillna(''))
    # Append human-readable category words
    corpus_parts.append(
        df['category_id'].apply(lambda x: CAT_NAME_MAP.get(int(x), '') if pd.notna(x) else '')
    )
    corpus = corpus_parts[0]
    for part in corpus_parts[1:]:
        corpus = corpus + " " + part
    corpus = corpus.str.lower()

    tokens = [t.strip() for t in query.lower().split() if t.strip()]
    if not tokens:
        return pd.DataFrame()

    # AND across tokens — every token must appear somewhere in corpus
    mask = pd.Series([True] * len(df), index=df.index)
    for token in tokens:
        mask = mask & corpus.str.contains(token, regex=False, na=False)

    results = df[mask].drop_duplicates('product_id')

    # If AND gives nothing, fall back to OR (any token matches)
    if results.empty and len(tokens) > 1:
        mask_or = pd.Series([False] * len(df), index=df.index)
        for token in tokens:
            mask_or = mask_or | corpus.str.contains(token, regex=False, na=False)
        results = df[mask_or].drop_duplicates('product_id')

    return results.head(max_results)


search_examples = get_search_examples(df_products, n=4)
placeholder_text = "e.g. " + ",  ".join(search_examples)

st.markdown('<div class="section-heading">🔍 Search</div>', unsafe_allow_html=True)

# Subheading explains what's searchable
searchable_cols = ["product name", "brand"]
if 'color' in df_products.columns:      searchable_cols.append("colour")
if 'style_attributes' in df_products.columns: searchable_cols.append("style")
st.markdown(
    f'<div class="section-sub">Search by {" · ".join(searchable_cols)} · or category (dresses / jeans)</div>',
    unsafe_allow_html=True
)

search_query = st.text_input(
    "search",
    placeholder=placeholder_text,
    label_visibility="collapsed"
)

if search_query.strip():
    raw_results = robust_search(df_products, search_query.strip(), max_results=12)
    results = apply_filters(raw_results, selected_cats, selected_brands, price_range)

    if not results.empty:
        # Show how many found + which tokens matched
        tokens_used = [t for t in search_query.lower().split() if t.strip()]
        st.markdown(
            f'<div class="section-sub">{len(results)} results for &ldquo;{search_query}&rdquo;</div>',
            unsafe_allow_html=True
        )
        s_cols = st.columns(min(len(results), 6))
        for i, (_, row) in enumerate(results.head(6).iterrows()):
            with s_cols[i]:
                render_card(
                    row['local_image_path'], row['product_name'],
                    row['category_id'], row.get('pdp_url', None)
                )
                wishlist_button(
                    row['product_id'],
                    row['product_name'],
                    row.get('pdp_url', None),
                    key_suffix=f"search_{i}"
                )
    else:
        # Helpful no-results message with suggestions
        st.markdown(f"""
        <div style="padding:1.2rem 1.5rem;background:var(--card-bg);
                    border:1px solid var(--card-border);border-radius:var(--radius);
                    font-size:.8rem;color:var(--mist);letter-spacing:.05em;">
            No results for <span style="color:var(--gold);">"{search_query}"</span>.
            Try: <span style="color:var(--gold-lt);">dresses · jeans · {" · ".join(search_examples[:2])}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Shop by Style ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">✦ Shop by Style</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Browse dresses or jeans — or let us surprise you</div>', unsafe_allow_html=True)

style_col1, style_col2, style_col3 = st.columns([1, 1, 1])

with style_col1:
    if st.button("👗  Dresses", key="style_dresses", use_container_width=True):
        st.session_state.style_filter = "dresses"
        st.rerun()
with style_col2:
    if st.button("👖  Jeans", key="style_jeans", use_container_width=True):
        st.session_state.style_filter = "jeans"
        st.rerun()
with style_col3:
    if st.button("🎲  Surprise Me", key="style_surprise", use_container_width=True):
        import random
        st.session_state.style_filter  = "surprise"
        st.session_state.style_seed    = random.randint(0, 9999)
        st.rerun()

if 'style_filter' not in st.session_state:
    st.session_state.style_filter = None

if st.session_state.style_filter == "dresses":
    pool = df_products[df_products['category_id'] == CAT_DRESSES].drop_duplicates('product_id')
    pool = apply_filters(pool, [], selected_brands, price_range)
    label = "👗 Dresses"
elif st.session_state.style_filter == "jeans":
    pool = df_products[df_products['category_id'] == CAT_JEANS].drop_duplicates('product_id')
    pool = apply_filters(pool, [], selected_brands, price_range)
    label = "👖 Jeans"
elif st.session_state.style_filter == "surprise":
    pool = df_products.drop_duplicates('product_id')
    pool = apply_filters(pool, [], selected_brands, price_range)
    label = "🎲 Surprise Pick"
else:
    pool = pd.DataFrame()
    label = ""

if not pool.empty:
    st.markdown(
        f'<div style="font-family:\'Cormorant Garamond\',serif;font-size:1rem;'
        f'color:var(--gold-lt);letter-spacing:.1em;margin:.6rem 0 1rem;">— {label}</div>',
        unsafe_allow_html=True
    )
    sample = pool.sample(min(6, len(pool)), random_state=st.session_state.style_seed)
    s_cols = st.columns(min(len(sample), 6))
    for i, (_, row) in enumerate(sample.iterrows()):
        with s_cols[i]:
            render_card(
                row['local_image_path'], row['product_name'],
                row['category_id'], row.get('pdp_url', None)
            )
            wishlist_button(
                row['product_id'],
                row['product_name'],
                row.get('pdp_url', None),
                key_suffix=f"style_{i}"
            )
elif st.session_state.style_filter:
    st.info("No products found for this style.")

# ─── Upload Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">✦ Discover Your Look</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Upload any fashion item — we do the rest</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your fashion image here — JPG, PNG, WebP supported",
    type=None,
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
        fname     = uploaded_file.name.lower()

        def extract_image_from_mhtml(data: bytes):
            import email as _email, re, base64 as _b64
            candidates = []
            try:
                msg = _email.message_from_bytes(data)
                for part in msg.walk():
                    if part.get_content_type().startswith('image/'):
                        payload = part.get_payload(decode=True)
                        if payload and len(payload) > 200:
                            candidates.append(payload)
            except Exception:
                pass
            if not candidates:
                try:
                    msg = _email.message_from_string(data.decode('latin-1'))
                    for part in msg.walk():
                        if part.get_content_type().startswith('image/'):
                            payload = part.get_payload(decode=True)
                            if payload and len(payload) > 200:
                                candidates.append(payload)
                except Exception:
                    pass
            try:
                pattern = rb'Content-Type:\s*image/[^\r\n]+[\r\n]+(?:Content-[^\r\n]+[\r\n]+)*[\r\n]+((?:[A-Za-z0-9+/=]+[\r\n]+)+)'
                for m in re.findall(pattern, data, re.IGNORECASE):
                    clean = re.sub(rb'\s+', b'', m)
                    if len(clean) > 200:
                        try:
                            decoded = _b64.b64decode(clean + b'==')
                            if len(decoded) > 200:
                                candidates.append(decoded)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                for m in re.finditer(rb'data:image/[^;]+;base64,([A-Za-z0-9+/=\r\n]+)', data):
                    clean = re.sub(rb'\s+', b'', m.group(1))
                    if len(clean) > 200:
                        try:
                            decoded = _b64.b64decode(clean + b'==')
                            if len(decoded) > 200:
                                candidates.append(decoded)
                        except Exception:
                            pass
            except Exception:
                pass
            if not candidates:
                return None
            return max(candidates, key=len)

        if fname.endswith(('.mhtml', '.mht')):
            img_bytes = extract_image_from_mhtml(raw_bytes)
            if img_bytes:
                raw_bytes = img_bytes
            else:
                st.error("❌ Could not extract any image from MHTML file.")
                st.stop()

        try:
            uploaded_image = Image.open(BytesIO(raw_bytes)).convert('RGB')
        except Exception as pil_err:
            extracted = extract_image_from_mhtml(raw_bytes)
            if extracted:
                try:
                    uploaded_image = Image.open(BytesIO(extracted)).convert('RGB')
                except Exception:
                    st.error("❌ Found image data but could not decode it.")
                    st.stop()
            else:
                st.error(f"❌ Could not open file. PIL error: `{pil_err}`")
                st.stop()

        _, mid_col, _ = st.columns([1, 1, 1])
        with mid_col:
            st.markdown('<div class="upload-frame">', unsafe_allow_html=True)
            st.image(uploaded_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Feature extraction ────────────────────────────────────────────────
        with st.spinner("✦  Analysing your item..."):
            query_features = extract_query_features(uploaded_image, feature_extractor_model, preprocess, device)

        if query_features is not None:
            D, I = perform_visual_search(query_features, faiss_index, k=6)

            if I is not None:
                # ── Similar products ──────────────────────────────────────────
                st.markdown('<div class="section-heading">◈ Visually Similar</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Pieces curated by visual intelligence</div>', unsafe_allow_html=True)

                similar_product_indices    = I[0]
                unique_similar_product_ids = []
                top_similar_product_id_for_history = None
                similarity_scores = {}

                for rank, idx in enumerate(similar_product_indices):
                    prod_id = all_product_ids[idx]
                    dist    = float(D[0][rank])
                    match_pct = max(0, int(100 - dist * 10))

                    if top_similar_product_id_for_history is None:
                        top_similar_product_id_for_history = prod_id
                    if prod_id not in unique_similar_product_ids and (idx != similar_product_indices[0] or D[0][0] > 0.001):
                        unique_similar_product_ids.append(prod_id)
                        similarity_scores[prod_id] = match_pct
                    if len(unique_similar_product_ids) >= 5:
                        break

                display_products_df = df_products[df_products['product_id'].isin(unique_similar_product_ids)].copy()
                display_products_df = apply_filters(display_products_df, selected_cats, selected_brands, price_range)

                if not display_products_df.empty:
                    cols = st.columns(min(len(display_products_df), 5))
                    for i, (_, row) in enumerate(display_products_df.head(5).iterrows()):
                        with cols[i]:
                            pid   = row['product_id']
                            score = similarity_scores.get(pid, None)
                            render_card(
                                row['local_image_path'], row['product_name'],
                                row['category_id'], row.get('pdp_url'), match_score=score
                            )
                            wishlist_button(
                                row['product_id'],
                                row['product_name'],
                                row.get('pdp_url', None),
                                key_suffix=f"sim_{i}"
                            )
                            already_saved = pid in st.session_state.wishlist
                else:
                    st.info("No similar products found with current filters.")

                # ── Share outfit ──────────────────────────────────────────────
                if unique_similar_product_ids:
                    share_ids   = ",".join(unique_similar_product_ids[:5])
                    outfit_code = base64.urlsafe_b64encode(share_ids.encode()).decode().rstrip("=")
                    share_url   = f"?outfit={outfit_code}"

                    st.markdown(f"""
                    <div class="share-box">
                        <div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:1rem;color:var(--gold-lt);">
                                🔗 Share this Outfit
                            </div>
                            <div style="font-size:.7rem;color:var(--mist);letter-spacing:.08em;margin-top:.2rem;">
                                Copy link and share with friends
                            </div>
                        </div>
                        <div class="share-code">vastra.app/{share_url}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Update history ────────────────────────────────────────────
                if top_similar_product_id_for_history:
                    if top_similar_product_id_for_history not in st.session_state.search_history:
                        st.session_state.search_history.append(top_similar_product_id_for_history)
                    if len(st.session_state.search_history) > 5:
                        st.session_state.search_history = st.session_state.search_history[-5:]

                # ── Complete the Look (direct opposite-category logic) ─────────
                st.markdown('<div class="section-heading">◉ Complete the Look</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Pair it with these picks</div>', unsafe_allow_html=True)

                # Find the uploaded image's closest match category
                top_pid      = str(all_product_ids[I[0][0]])
                top_row_df   = df_products[df_products['product_id'] == top_pid]
                top_cat      = int(top_row_df.iloc[0]['category_id']) if not top_row_df.empty else None

                # Opposite category: Dress→Jeans, Jeans→Dress
                OPPOSITE_CAT = {CAT_DRESSES: CAT_JEANS, CAT_JEANS: CAT_DRESSES}
                comp_cat     = OPPOSITE_CAT.get(top_cat)
                CAT_LABELS   = {CAT_DRESSES: "Dresses", CAT_JEANS: "Jeans"}

                if comp_cat:
                    comp_pool = df_products[
                        (df_products['category_id'] == comp_cat) &
                        (~df_products['product_id'].isin(unique_similar_product_ids + [top_pid]))
                    ].drop_duplicates('product_id')

                    if not comp_pool.empty:
                        comp_sample  = comp_pool.sample(min(4, len(comp_pool)), random_state=7)
                        all_rec_ids  = comp_sample['product_id'].tolist()
                        outfit_score = calculate_outfit_score(top_pid, all_rec_ids, df_products)

                        st.markdown(f"""
                        <div class="score-bar-wrap">
                            <div class="score-number">{outfit_score}%</div>
                            <div>
                                <div class="score-label">Outfit Match Score</div>
                                <div class="score-track">
                                    <div class="score-fill" style="width:{outfit_score}%;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="font-family:'Cormorant Garamond',serif;font-size:1.05rem;
                                    color:var(--gold-lt);letter-spacing:.08em;margin:.8rem 0 .8rem;">
                            — Pair with {CAT_LABELS.get(comp_cat, 'Complementary')}
                        </div>""", unsafe_allow_html=True)

                        rec_cols = st.columns(len(comp_sample))
                        for j, (_, item) in enumerate(comp_sample.iterrows()):
                            with rec_cols[j]:
                                render_card(
                                    item['local_image_path'], item['product_name'],
                                    item['category_id'], item.get('pdp_url', None)
                                )
                                rec_pid = item['product_id']
                                already_saved = rec_pid in st.session_state.wishlist
                                btn_label = "✓ Saved" if already_saved else "🔖 Save"
                                if st.button(btn_label, key=f"wl_rec_{rec_pid}_{j}",
                                             use_container_width=True):
                                    if rec_pid not in st.session_state.wishlist:
                                        st.session_state.wishlist[rec_pid] = {
                                            'name':    item['product_name'],
                                            'pdp_url': str(item.get('pdp_url', ''))
                                        }
                                        st.toast("Added to wishlist!", icon="🔖")
                                        st.rerun()
                    else:
                        st.info("No complementary pieces available right now.")
                else:
                    st.info("Upload a dress or jeans to see outfit pairings.")

                # ── Personalised edits ────────────────────────────────────────
                if len(st.session_state.search_history) > 1:
                    st.markdown('<div class="section-heading">⟡ Your Curated Edits</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-sub">Based on your browsing taste</div>', unsafe_allow_html=True)

                    past_ids = list(st.session_state.search_history[:-1])
                    past_df  = df_products[df_products['product_id'].isin(past_ids)].copy()

                    if not past_df.empty:
                        past_cats   = past_df['category_id'].unique()
                        past_brands = past_df['brand'].unique() if 'brand' in past_df.columns else []
                        excluded    = st.session_state.search_history + unique_similar_product_ids

                        suggestions = df_products[
                            (df_products['category_id'].isin(past_cats)) |
                            (df_products['brand'].isin(past_brands) if len(past_brands) > 0 else False)
                        ].copy()
                        suggestions = suggestions[~suggestions['product_id'].isin(excluded)]
                        suggestions = apply_filters(suggestions, selected_cats, selected_brands, price_range)

                        if not suggestions.empty:
                            sample = suggestions.sample(min(6, len(suggestions)), random_state=43)
                            p_cols = st.columns(len(sample))
                            for i, (_, row) in enumerate(sample.iterrows()):
                                with p_cols[i]:
                                    render_card(
                                        row['local_image_path'], row['product_name'],
                                        row['category_id'], row.get('pdp_url', None)
                                    )
                                    wishlist_button(
                                        row['product_id'],
                                        row['product_name'],
                                        row.get('pdp_url', None),
                                        key_suffix=f"personal_{i}"
                                    )
                        else:
                            st.info("No new personalised picks with current filters.")
                    else:
                        st.info("Search history items not found in database.")
                else:
                    st.markdown("""
                    <div style="margin-top:1.5rem;padding:1.2rem 1.5rem;background:var(--card-bg);
                                border:1px solid var(--card-border);border-radius:var(--radius);
                                font-size:0.8rem;color:var(--mist);letter-spacing:.08em;">
                        ✦ &nbsp; Search a few more items to unlock your personalised edits.
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.error("Could not extract features from the uploaded image.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

else:
    st.markdown("""
    <div style="margin:1rem 0 2rem;padding:3rem;background:var(--card-bg);
                border:1.5px dashed rgba(201,168,76,0.15);border-radius:var(--radius);
                text-align:center;color:var(--mist);">
        <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                    color:var(--gold);opacity:.4;margin-bottom:.6rem;">↑</div>
        <div style="font-size:.75rem;letter-spacing:.2em;text-transform:uppercase;opacity:.5;">
            Upload a fashion item to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Trending / New Arrivals ───────────────────────────────────────────────────
st.markdown("""
<div class="section-heading">
    ✦ New Arrivals
    <span class="trending-badge">Live</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-sub">The freshest drops in the catalogue</div>', unsafe_allow_html=True)

if trending_products_df is not None and not trending_products_df.empty:
    filtered_trending = apply_filters(
        df_products[df_products['product_id'].isin(trending_products_df['product_id'])],
        selected_cats, selected_brands, price_range
    ).head(6)

    if not filtered_trending.empty:
        t_cols = st.columns(min(len(filtered_trending), 6))
        for i, (_, row) in enumerate(filtered_trending.iterrows()):
            with t_cols[i]:
                render_card(
                    row['local_image_path'], row['product_name'],
                    row['category_id'], row.get('pdp_url', None)
                )
                wishlist_button(
                    row['product_id'],
                    row['product_name'],
                    row.get('pdp_url', None),
                    key_suffix=f"trend_{i}"
                )
    else:
        st.info("No new arrivals match current filters.")
else:
    st.info("No new arrivals found in the dataset.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding:2rem 0;border-top:1px solid rgba(201,168,76,0.12);
            text-align:center;color:var(--mist);opacity:.4;">
    <span style="font-family:'Cormorant Garamond',serif;font-size:1rem;letter-spacing:.3em;">
        VASTRA
    </span>
    <span style="display:block;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;
                 margin-top:.3rem;">
        Powered by ResNet50 · Faiss · Streamlit
    </span>
</div>
""", unsafe_allow_html=True)