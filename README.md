# Product Recommendation System

A multi-modal product recommendation system that uses both text and image embeddings to find similar products. Built with Streamlit for an interactive user experience.

## Features

- **Multiple Recommendation Modes:**
  - Product Name Only
  - Product Name + Description
  - Product Image Only
  - Name + Description + Image (Multi-modal)
  - Product Name + Image

- **Parametric Weighting:**
  - Adjustable text and image weights for combined modes
  - Real-time weight visualization

- **State-of-the-Art Models:**
  - Text: Multilingual MiniLM-L12-v2 - Fast multilingual support (including Turkish)
  - Image: ResNet50 pre-trained on ImageNet

- **Interactive UI:**
  - Product selection and preview
  - Similarity scores
  - Configurable number of recommendations

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your data is properly structured:
   - `ürün.csv`: CSV file with columns: `id`, `product_name`, `product_description`
   - `images/`: Directory containing product images named as `{id}.jpg` or `{id}.jpeg`

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open your browser and navigate to the provided local URL (usually `http://localhost:8501`)

## How It Works

### Text Embeddings
The system uses Multilingual MiniLM-L12-v2 to convert product names and descriptions into dense vector representations. This lightweight model is specifically designed to work with multiple languages, including Turkish, and offers fast inference times.

### Image Embeddings
Product images are processed using ResNet50, a deep convolutional neural network pre-trained on ImageNet. The final classification layer is removed to extract 2048-dimensional feature vectors.

### Similarity Calculation
- **Single Mode:** Cosine similarity is calculated directly between embeddings
- **Combined Mode:** Weighted average of text and image similarities:
  ```
  final_similarity = (w_text × text_similarity) + (w_image × image_similarity)
  ```

### Recommendation Process
1. Select a product
2. Choose recommendation mode
3. Adjust weights (for combined modes)
4. System calculates similarity scores with all other products
5. Top N most similar products are returned, ranked by similarity score

## Project Structure

```
rec/
├── app.py                      # Streamlit web application
├── recommendation_engine.py    # Core recommendation logic
├── requirements.txt            # Python dependencies
├── ürün.csv                   # Product data
├── images/                     # Product images
│   ├── 1.jpeg
│   ├── 2.jpg
│   └── ...
└── README.md                   # This file
```

## Configuration Options

### Sidebar Controls
- **Product Selection:** Choose a product to find recommendations for
- **Recommendation Mode:** Select which features to use for similarity
- **Number of Recommendations:** How many similar products to return (1-9)
- **Text Weight:** Importance of text similarity (0.0-1.0)
- **Image Weight:** Importance of image similarity (0.0-1.0)

### Recommendation Modes

1. **Product Name Only** (`name`)
   - Uses only the product name field
   - Best for finding products with similar names

2. **Product Name + Description** (`name_desc`)
   - Combines product name and full description
   - Best for semantic similarity based on features and characteristics

3. **Product Image Only** (`image`)
   - Uses only visual features
   - Best for finding visually similar products (color, style, shape)

4. **Name + Description + Image** (`text_image`)
   - Combines all available information
   - Most comprehensive similarity measure
   - Weights can be adjusted to prioritize text or visual similarity

5. **Product Name + Image** (`name_image`)
   - Combines product name with visual features
   - Good balance between semantic and visual similarity

## Technical Details

### Models
- **Multilingual MiniLM-L12-v2:** 384-dimensional multilingual sentence embeddings (Türkçe destekli)
- **ResNet50:** 2048-dimensional image feature vectors

### Performance
- **First run:** Models are downloaded (~500MB) and embeddings are generated (~30 seconds)
- **Second run onwards:** Embeddings are loaded from cache in 1-2 seconds! ⚡
- **Auto-regeneration:** If CSV file changes, embeddings are automatically regenerated
- GPU acceleration supported if available (CPU fallback included)

### Caching System
- **Models:** Downloaded once to `~/.cache/huggingface/` and `~/.cache/torch/` (persistent across projects)
- **Embeddings:** Saved to `embeddings_cache.pkl` (regenerated only when CSV changes)
- **Streamlit:** `@cache_resource` decorator keeps engine in memory during the session

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- ~2GB disk space for model downloads
- ~4GB RAM minimum

## Troubleshooting

**Issue:** Models not downloading
- Check your internet connection
- Try running with `--no-cache-dir` flag during pip install

**Issue:** Out of memory error
- Reduce batch size in embedding generation
- Use CPU instead of GPU
- Close other applications

**Issue:** Image not loading
- Verify image files exist in the `images/` directory
- Check file naming matches product IDs
- Supported formats: JPEG, JPG, PNG

## Future Enhancements

- [ ] Add batch recommendation support
- [ ] Implement A/B testing for different weighting strategies
- [ ] Add user feedback mechanism
- [ ] Support for more languages and image formats
- [ ] Export recommendations to CSV
- [ ] Add filtering options (price range, category, etc.)

## License

This project is provided as-is for educational and commercial use.

