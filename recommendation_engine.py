import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle
import os
from datetime import datetime


class ProductRecommendationEngine:
    """
    Multi-modal product recommendation system using text and image embeddings.
    Supports various recommendation modes with parametric weighting.
    """
    
    def __init__(self, csv_path: str, images_dir: str, cache_file: str = "embeddings_cache.pkl"):
        """
        Initialize the recommendation engine.
        
        Args:
            csv_path: Path to the CSV file containing product data
            images_dir: Directory containing product images
            cache_file: Path to the cache file for embeddings
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.cache_file = cache_file
        
        # Load product data
        self.products_df = pd.read_csv(csv_path)
        
        # Initialize models
        print("Loading text embedding model (Multilingual MiniLM)...")
        self.text_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        print("Loading image embedding model (ResNet50)...")
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer to get embeddings
        self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
        self.image_model.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Generate or load embeddings
        self.text_embeddings = None
        self.image_embeddings = None
        self._load_or_generate_embeddings()
    
    def _check_cache_validity(self) -> bool:
        """
        Check if the cache file exists and is valid.
        
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(self.cache_file):
            print("No cache file found. Will generate new embeddings.")
            return False
        
        # Check if CSV file has been modified after cache was created
        cache_mtime = os.path.getmtime(self.cache_file)
        csv_mtime = os.path.getmtime(self.csv_path)
        
        if csv_mtime > cache_mtime:
            print("CSV file has been modified. Will regenerate embeddings.")
            return False
        
        print("Valid cache file found!")
        return True
    
    def _load_or_generate_embeddings(self):
        """Load embeddings from cache or generate new ones."""
        if self._check_cache_validity():
            try:
                print("Loading embeddings from cache...")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.text_embeddings = cache_data['text_embeddings']
                self.image_embeddings = cache_data['image_embeddings']
                
                # Verify that the number of products matches
                if len(self.text_embeddings) == len(self.products_df):
                    print(f"✓ Successfully loaded embeddings for {len(self.products_df)} products from cache!")
                    return
                else:
                    print("Cache size mismatch. Will regenerate embeddings.")
            except Exception as e:
                print(f"Error loading cache: {e}. Will regenerate embeddings.")
        
        # Generate new embeddings
        self._generate_embeddings()
        
        # Save to cache
        self._save_embeddings_to_cache()
    
    def _generate_embeddings(self):
        """Generate all text and image embeddings for products."""
        print("Generating text embeddings...")
        # Combine product name and description for text embeddings
        texts = []
        for _, row in self.products_df.iterrows():
            text = f"{row['product_name']} {row['product_description']}"
            texts.append(text)
        
        self.text_embeddings = self.text_model.encode(texts, show_progress_bar=True)
        
        print("Generating image embeddings...")
        image_embeddings_list = []
        for product_id in self.products_df['id']:
            image_path = self._get_image_path(product_id)
            if image_path and image_path.exists():
                embedding = self._get_image_embedding(image_path)
                image_embeddings_list.append(embedding)
            else:
                # If image not found, use zero vector
                image_embeddings_list.append(np.zeros(2048))
        
        self.image_embeddings = np.array(image_embeddings_list)
        print("Embeddings generated successfully!")
    
    def _save_embeddings_to_cache(self):
        """Save generated embeddings to cache file."""
        try:
            print("Saving embeddings to cache...")
            cache_data = {
                'text_embeddings': self.text_embeddings,
                'image_embeddings': self.image_embeddings,
                'num_products': len(self.products_df),
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"✓ Embeddings saved to {self.cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_image_path(self, product_id: int) -> Optional[Path]:
        """Get the image path for a product ID."""
        # Try different extensions
        for ext in ['.jpeg', '.jpg', '.png']:
            path = self.images_dir / f"{product_id}{ext}"
            if path.exists():
                return path
        return None
    
    def _get_image_embedding(self, image_path: Path) -> np.ndarray:
        """Get embedding for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.image_model(image_tensor)
            
            return embedding.squeeze().numpy()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(2048)
    
    def get_recommendations(
        self,
        product_id: int,
        mode: str = "text_image",
        n_recommendations: int = 5,
        text_weight: float = 0.5,
        image_weight: float = 0.5
    ) -> List[Dict]:
        """
        Get product recommendations based on selected mode.
        
        Args:
            product_id: ID of the product to find recommendations for
            mode: One of ['name', 'name_desc', 'image', 'text_image', 'name_image']
            n_recommendations: Number of recommendations to return
            text_weight: Weight for text similarity (used in combined modes)
            image_weight: Weight for image similarity (used in combined modes)
            
        Returns:
            List of recommended products with similarity scores
        """
        # Get product index
        try:
            product_idx = self.products_df[self.products_df['id'] == product_id].index[0]
        except IndexError:
            raise ValueError(f"Product ID {product_id} not found")
        
        similarities = None
        
        if mode == "name":
            # Use only product name
            texts = self.products_df['product_name'].tolist()
            embeddings = self.text_model.encode(texts)
            similarities = cosine_similarity([embeddings[product_idx]], embeddings)[0]
            
        elif mode == "name_desc":
            # Use product name + description (already computed)
            similarities = cosine_similarity(
                [self.text_embeddings[product_idx]], 
                self.text_embeddings
            )[0]
            
        elif mode == "image":
            # Use only image
            similarities = cosine_similarity(
                [self.image_embeddings[product_idx]], 
                self.image_embeddings
            )[0]
            
        elif mode == "text_image":
            # Combine name+description and image
            text_sim = cosine_similarity(
                [self.text_embeddings[product_idx]], 
                self.text_embeddings
            )[0]
            image_sim = cosine_similarity(
                [self.image_embeddings[product_idx]], 
                self.image_embeddings
            )[0]
            similarities = text_weight * text_sim + image_weight * image_sim
            
        elif mode == "name_image":
            # Combine only name and image
            texts = self.products_df['product_name'].tolist()
            name_embeddings = self.text_model.encode(texts)
            text_sim = cosine_similarity(
                [name_embeddings[product_idx]], 
                name_embeddings
            )[0]
            image_sim = cosine_similarity(
                [self.image_embeddings[product_idx]], 
                self.image_embeddings
            )[0]
            similarities = text_weight * text_sim + image_weight * image_sim
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Get top N similar products (excluding the product itself)
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if idx != product_idx and len(recommendations) < n_recommendations:
                product_data = self.products_df.iloc[idx]
                recommendations.append({
                    'id': int(product_data['id']),
                    'name': product_data['product_name'],
                    'description': product_data['product_description'],
                    'similarity_score': float(similarities[idx]),
                    'image_path': str(self._get_image_path(int(product_data['id'])))
                })
        
        return recommendations
    
    def get_product_details(self, product_id: int) -> Dict:
        """Get details for a specific product."""
        try:
            product_data = self.products_df[self.products_df['id'] == product_id].iloc[0]
            return {
                'id': int(product_data['id']),
                'name': product_data['product_name'],
                'description': product_data['product_description'],
                'image_path': str(self._get_image_path(int(product_data['id'])))
            }
        except IndexError:
            raise ValueError(f"Product ID {product_id} not found")
    
    def get_all_products(self) -> List[Dict]:
        """Get all products."""
        products = []
        for _, row in self.products_df.iterrows():
            products.append({
                'id': int(row['id']),
                'name': row['product_name'],
                'description': row['product_description'],
                'image_path': str(self._get_image_path(int(row['id'])))
            })
        return products

