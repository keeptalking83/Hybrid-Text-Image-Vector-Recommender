import streamlit as st
from recommendation_engine import ProductRecommendationEngine
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .similarity-score {
        font-weight: bold;
        color: #2ecc71;
        font-size: 1.2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize the recommendation engine (cached)
@st.cache_resource
def load_recommendation_engine():
    """Load and cache the recommendation engine."""
    csv_path = "ürün.csv"
    images_dir = "images"
    return ProductRecommendationEngine(csv_path, images_dir)


def display_product_card(product, show_similarity=False):
    """Display a product card with image and details."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if product['image_path'] and os.path.exists(product['image_path']):
            try:
                image = Image.open(product['image_path'])
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.warning("No image available")
    
    with col2:
        st.markdown(f"### {product['name']}")
        st.write(product['description'])
        
        if show_similarity and 'similarity_score' in product:
            st.markdown(
                f"<p class='similarity-score'>Similarity Score: {product['similarity_score']:.4f}</p>",
                unsafe_allow_html=True
            )


def main():
    st.title("🛍️ Product Recommendation System")
    st.markdown("---")
    
    # Load recommendation engine
    with st.spinner("Loading recommendation engine..."):
        try:
            engine = load_recommendation_engine()
        except Exception as e:
            st.error(f"Error loading recommendation engine: {e}")
            return
    
    # Get all products
    all_products = engine.get_all_products()
    
    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Product selection
    product_names = [f"{p['id']} - {p['name']}" for p in all_products]
    selected_product_str = st.sidebar.selectbox(
        "Select a product:",
        product_names
    )
    selected_product_id = int(selected_product_str.split(" - ")[0])
    
    # Recommendation mode
    st.sidebar.markdown("### Recommendation Mode")
    mode_options = {
        "Product Name Only": "name",
        "Product Name + Description": "name_desc",
        "Product Image Only": "image",
        "Name + Description + Image": "text_image",
        "Product Name + Image": "name_image"
    }
    
    selected_mode_name = st.sidebar.radio(
        "Select similarity mode:",
        list(mode_options.keys())
    )
    selected_mode = mode_options[selected_mode_name]
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=9,
        value=5
    )
    
    # Weighting parameters (only for combined modes)
    if selected_mode in ["text_image", "name_image"]:
        st.sidebar.markdown("### Weighting Parameters")
        st.sidebar.info("Adjust the weights for text and image similarity. The sum does not need to equal 1.0.")
        
        text_weight = st.sidebar.slider(
            "Text Weight (w_text):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        image_weight = st.sidebar.slider(
            "Image Weight (w_image):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Display normalized weights
        total_weight = text_weight + image_weight
        if total_weight > 0:
            norm_text = text_weight / total_weight
            norm_image = image_weight / total_weight
            st.sidebar.markdown(f"**Normalized:** Text: {norm_text:.2f}, Image: {norm_image:.2f}")
    else:
        text_weight = 0.5
        image_weight = 0.5
    
    # Main content area
    col_selected, col_recommendations = st.columns([1, 2])
    
    with col_selected:
        st.header("Selected Product")
        selected_product = engine.get_product_details(selected_product_id)
        display_product_card(selected_product, show_similarity=False)
    
    with col_recommendations:
        st.header("Recommended Products")
        
        # Get recommendations button
        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Finding similar products..."):
                try:
                    recommendations = engine.get_recommendations(
                        product_id=selected_product_id,
                        mode=selected_mode,
                        n_recommendations=n_recommendations,
                        text_weight=text_weight,
                        image_weight=image_weight
                    )
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} similar products!")
                        
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"#### Recommendation #{i}")
                            display_product_card(rec, show_similarity=True)
                            st.markdown("---")
                    else:
                        st.warning("No recommendations found.")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
    
    # Information section
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### Multi-Modal Product Recommendation System
        
        This system uses state-of-the-art machine learning models to find similar products based on:
        
        **Text Embeddings:**
        - Model: Multilingual MiniLM-L12-v2
        - Supports multilingual text understanding (including Turkish)
        - Captures semantic meaning of product names and descriptions
        
        **Image Embeddings:**
        - Model: ResNet50 (pre-trained on ImageNet)
        - Extracts visual features from product images
        - Robust to variations in lighting, angle, and style
        
        **Recommendation Modes:**
        1. **Product Name Only:** Uses only the product name for similarity
        2. **Product Name + Description:** Combines name and description text
        3. **Product Image Only:** Uses only visual features
        4. **Name + Description + Image:** Combines all text and visual features
        5. **Product Name + Image:** Combines name and visual features
        
        **Similarity Calculation:**
        - Cosine similarity is used to measure similarity between embeddings
        - For combined modes, weighted average of text and image similarities
        - Scores range from 0 (completely different) to 1 (identical)
        """)


if __name__ == "__main__":
    main()

