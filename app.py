from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import os
import logging
from datetime import datetime

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables untuk store trained models
content_vectorizer = None
content_tfidf_matrix = None
content_products_df = None
collaborative_model = None
collaborative_products_df = None
is_content_trained = False
is_collaborative_trained = False

def convert_quantity_to_rating(quantity):
    """Convert quantity to rating scale 1-5"""
    if quantity >= 10:
        return 5
    elif quantity >= 7:
        return 4
    elif quantity >= 5:
        return 3
    elif quantity >= 3:
        return 2
    else:
        return 1

def train_content_based(products_data):
    """Train content-based model"""
    global content_vectorizer, content_tfidf_matrix, content_products_df, is_content_trained
    
    try:
        # Convert products data to DataFrame
        products_list = []
        for product in products_data:
            umkm_name = product.get('UMKM', {}).get('nama', '') if product.get('UMKM') else ''
            kategori_name = product.get('kategori', {}).get('nama', '')
            
            products_list.append({
                'id': product['id'],
                'nama': product['nama'],
                'harga': product['harga'],
                'gambar': product['gambar'],
                'stok': product['stok'],
                'status': product['status'],
                'kategori_name': kategori_name,
                'umkm_name': umkm_name,
                'UMKM': product.get('UMKM'),
                'kategori': product.get('kategori'),
                'ProdukVarian': product.get('ProdukVarian', [])
            })
        
        content_products_df = pd.DataFrame(products_list)
        
        if content_products_df.empty:
            logger.error("No products data provided for content-based")
            return False
        
        # Create feature text: combine nama, kategori, and umkm
        features = (
            content_products_df['nama'].fillna('') + ' ' + 
            content_products_df['kategori_name'].fillna('') + ' ' + 
            content_products_df['umkm_name'].fillna('')
        )
        
        # Initialize vectorizer and create TF-IDF matrix
        content_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000, 
            ngram_range=(1, 2),
            lowercase=True
        )
        content_tfidf_matrix = content_vectorizer.fit_transform(features)
        is_content_trained = True
        
        logger.info(f"Content-based model trained with {len(content_products_df)} products")
        return True
        
    except Exception as e:
        logger.error(f"Error training content-based model: {str(e)}")
        return False

def get_content_recommendations(cart_items, num_recommendations=5):
    """Get content-based recommendations"""
    global content_vectorizer, content_tfidf_matrix, content_products_df, is_content_trained
    
    if not is_content_trained:
        logger.warning("Content-based model not trained")
        return []
    
    try:
        # Debug logging
        logger.info(f"Getting content recommendations for {len(cart_items)} cart items")
        logger.info(f"Products DF shape: {content_products_df.shape if content_products_df is not None else 'None'}")
        
        # Get unique product IDs from cart
        cart_product_ids = list(set([item['id'] for item in cart_items]))
        logger.info(f"Cart product IDs: {cart_product_ids}")
        
        # Find indices of cart products
        cart_indices = []
        for product_id in cart_product_ids:
            idx = content_products_df[content_products_df['id'] == product_id].index
            if len(idx) > 0:
                cart_indices.append(idx[0])
                logger.info(f"Found product {product_id} at index {idx[0]}")
            else:
                logger.warning(f"Product {product_id} not found in trained data")
        
        if not cart_indices:
            logger.warning("No cart products found in trained data")
            return []
        
        # Check if we have enough products to recommend
        total_products = len(content_products_df)
        if total_products <= len(cart_indices):
            logger.warning(f"Not enough products to recommend. Total: {total_products}, Cart: {len(cart_indices)}")
            return []
        
        # Calculate mean similarity for all cart items
        all_similarities = []
        for cart_idx in cart_indices:
            # Calculate cosine similarity for this cart item
            cosine_sim = cosine_similarity(
                content_tfidf_matrix[cart_idx:cart_idx+1], 
                content_tfidf_matrix
            ).flatten()
            all_similarities.append(cosine_sim)
        
        # Average similarities across all cart items
        if len(all_similarities) > 1:
            avg_similarity = np.mean(all_similarities, axis=0)
        else:
            avg_similarity = all_similarities[0]
        
        # Get similarity scores with indices
        sim_scores = list(enumerate(avg_similarity))
        
        # Remove cart items from recommendations
        sim_scores = [score for score in sim_scores if score[0] not in cart_indices]
        
        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        sim_scores = sim_scores[:num_recommendations]
        
        # Build recommendations
        recommendations = []
        for idx, score in sim_scores:
            if idx < len(content_products_df):
                try:
                    product_data = content_products_df.iloc[idx]
                    
                    recommendation = {
                        'id': product_data['id'],
                        'nama': product_data['nama'],
                        'harga': int(product_data['harga']),
                        'gambar': product_data['gambar'],
                        'stok': int(product_data['stok']),
                        'status': bool(product_data['status']),
                        'score': float(score),
                        'UMKM': product_data['UMKM'],
                        'kategori': product_data['kategori'],
                        'ProdukVarian': product_data['ProdukVarian']
                    }
                    recommendations.append(recommendation)
                except Exception as item_error:
                    logger.error(f"Error processing recommendation item {idx}: {str(item_error)}")
                    continue
        
        logger.info(f"Generated {len(recommendations)} content-based recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting content recommendations: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def train_collaborative_filtering(transactions_data):
    """Train collaborative filtering model"""
    global collaborative_model, collaborative_products_df, is_collaborative_trained
    
    try:
        # Prepare transaction data for matrix factorization
        ratings_list = []
        product_info = {}
        
        for transaction in transactions_data:
            transaction_id = transaction['id']
            
            for item in transaction['transaksiItem']:
                product_id = item['id']  # This should be produkId from TransaksiItem
                quantity = item['jumlah']
                rating = convert_quantity_to_rating(quantity)
                
                ratings_list.append({
                    'user_id': transaction_id,
                    'item_id': product_id,
                    'rating': rating
                })
                
                # Store product info for later use
                if product_id not in product_info:
                    product_info[product_id] = {
                        'nama': item['produk']['nama']
                    }
        
        if not ratings_list:
            logger.error("No transaction data provided for collaborative filtering")
            return False
        
        # Create DataFrame
        ratings_df = pd.DataFrame(ratings_list)
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
        
        # Train SVD model
        collaborative_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        trainset = data.build_full_trainset()
        collaborative_model.fit(trainset)
        
        # Store product info
        collaborative_products_df = pd.DataFrame.from_dict(product_info, orient='index')
        collaborative_products_df.reset_index(inplace=True)
        collaborative_products_df.rename(columns={'index': 'product_id'}, inplace=True)
        
        is_collaborative_trained = True
        
        logger.info(f"Collaborative model trained with {len(ratings_list)} ratings from {len(set([r['user_id'] for r in ratings_list]))} transactions")
        return True
        
    except Exception as e:
        logger.error(f"Error training collaborative model: {str(e)}")
        return False

def get_collaborative_recommendations(cart_items, all_products, num_recommendations=5):
    """Get collaborative filtering recommendations"""
    global collaborative_model, collaborative_products_df, is_collaborative_trained
    
    if not is_collaborative_trained:
        return []
    
    try:
        # Get cart product IDs
        cart_product_ids = set([item['id'] for item in cart_items])
        
        # Create a dummy user ID for prediction
        dummy_user_id = "temp_user"
        
        # Get all available products
        all_product_ids = [p['id'] for p in all_products]
        
        # Predict ratings for all products
        predictions = []
        for product_id in all_product_ids:
            if product_id not in cart_product_ids:  # Don't recommend items already in cart
                pred = collaborative_model.predict(dummy_user_id, product_id)
                predictions.append((product_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_predictions = predictions[:num_recommendations]
        
        # Build recommendations with full product data
        recommendations = []
        for product_id, score in top_predictions:
            # Find product in all_products
            product_data = next((p for p in all_products if p['id'] == product_id), None)
            if product_data:
                recommendation = {
                    'id': product_data['id'],
                    'nama': product_data['nama'],
                    'harga': product_data['harga'],
                    'gambar': product_data['gambar'],
                    'stok': product_data['stok'],
                    'status': product_data['status'],
                    'score': float(score),
                    'UMKM': product_data.get('UMKM'),
                    'kategori': product_data.get('kategori'),
                    'ProdukVarian': product_data.get('ProdukVarian', [])
                }
                recommendations.append(recommendation)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting collaborative recommendations: {str(e)}")
        return []

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "hybrid-recommendation-system",
        "content_trained": is_content_trained,
        "collaborative_trained": is_collaborative_trained
    })

@app.route('/train-content', methods=['POST'])
def train_content_endpoint():
    """Train content-based model"""
    try:
        data = request.get_json()
        products_data = data.get('products', [])
        
        if not products_data:
            return jsonify({"error": "No products data provided"}), 400
        
        success = train_content_based(products_data)
        
        if success:
            return jsonify({
                "message": "Content-based model trained successfully",
                "total_products": len(products_data),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to train content-based model"}), 500
            
    except Exception as e:
        logger.error(f"Error in train-content endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train-collaborative', methods=['POST'])
def train_collaborative_endpoint():
    """Train collaborative filtering model"""
    try:
        data = request.get_json()
        transactions_data = data.get('transactions', [])
        
        if not transactions_data:
            return jsonify({"error": "No transactions data provided"}), 400
        
        success = train_collaborative_filtering(transactions_data)
        
        if success:
            return jsonify({
                "message": "Collaborative filtering model trained successfully",
                "total_transactions": len(transactions_data),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to train collaborative model"}), 500
            
    except Exception as e:
        logger.error(f"Error in train-collaborative endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend-content', methods=['POST'])
def recommend_content_endpoint():
    """Get content-based recommendations"""
    try:
        data = request.get_json()
        cart_items = data.get('items', [])
        num_recommendations = data.get('num_recommendations', 5)
        
        if not cart_items:
            return jsonify({"error": "No cart items provided"}), 400
        
        recommendations = get_content_recommendations(cart_items, num_recommendations)
        
        return jsonify({
            "recommendations": recommendations,
            "method": "content_based",
            "total_found": len(recommendations),
            "cart_items_count": len(cart_items)
        })
        
    except Exception as e:
        logger.error(f"Error in recommend-content endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend-collaborative', methods=['POST'])
def recommend_collaborative_endpoint():
    """Get collaborative filtering recommendations"""
    try:
        data = request.get_json()
        cart_items = data.get('items', [])
        all_products = data.get('all_products', [])
        num_recommendations = data.get('num_recommendations', 5)
        
        if not cart_items:
            return jsonify({"error": "No cart items provided"}), 400
        
        if not all_products:
            return jsonify({"error": "No products data provided"}), 400
        
        recommendations = get_collaborative_recommendations(cart_items, all_products, num_recommendations)
        
        return jsonify({
            "recommendations": recommendations,
            "method": "collaborative",
            "total_found": len(recommendations),
            "cart_items_count": len(cart_items)
        })
        
    except Exception as e:
        logger.error(f"Error in recommend-collaborative endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Hybrid Recommendation System API")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)