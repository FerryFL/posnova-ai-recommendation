from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Content Based Filtering Global Variable
is_content_trained = False
content_vectorizer = None
content_tfidf_matrix = None
content_products_df = None

# Apriori Global Variable
apriori_rules_df = None
apriori_products_df = None
is_apriori_trained = False

# def convert_quantity_to_rating(quantity):
#     """Convert quantity to rating scale 1-5"""
#     if quantity >= 10:
#         return 5
#     elif quantity >= 7:
#         return 4
#     elif quantity >= 5:
#         return 3
#     elif quantity >= 3:
#         return 2
#     else:
#         return 1

def train_content_based(products_data):
    """Train content-based model"""
    global content_vectorizer, content_tfidf_matrix, content_products_df, is_content_trained
    
    try:
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
        
        features = (
            content_products_df['nama'].fillna('') + ' ' + 
            content_products_df['kategori_name'].fillna('')
        )
        
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
        logger.info(f"Getting content recommendations for {len(cart_items)} cart items")
        logger.info(f"Products DF shape: {content_products_df.shape if content_products_df is not None else 'None'}")
        
        cart_product_ids = list(set([item['id'] for item in cart_items]))
        logger.info(f"Cart product IDs: {cart_product_ids}")
        
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
        
        total_products = len(content_products_df)
        if total_products <= len(cart_indices):
            logger.warning(f"Not enough products to recommend. Total: {total_products}, Cart: {len(cart_indices)}")
            return []
        
        all_similarities = []
        for cart_idx in cart_indices:
            text_sim = cosine_similarity(
                content_tfidf_matrix[cart_idx:cart_idx+1], 
                content_tfidf_matrix
            ).flatten()
            
            cart_price = content_products_df.iloc[cart_idx]['harga']
            all_prices = content_products_df['harga']

            cart_stock = content_products_df.iloc[cart_idx]['stok']
            all_stocks = content_products_df['stok']

            price_diff_pct = np.abs(all_prices - cart_price) / cart_price
            price_sim = 1 / (1 + price_diff_pct)

            stock_diff_pct = np.abs(all_stocks - cart_stock) / cart_stock
            stock_sim = 1 / (1 + stock_diff_pct)
            
            combine_sim = 0.6 * text_sim + 0.3 * price_sim + 0.1 * stock_sim
            all_similarities.append(combine_sim)
        
        if len(all_similarities) > 1:
            avg_similarity = np.mean(all_similarities, axis=0)
        else:
            avg_similarity = all_similarities[0]
        
        sim_scores = list(enumerate(avg_similarity))
        sim_scores = [score for score in sim_scores if score[0] not in cart_indices]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:num_recommendations]
        
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
                    if recommendation['score'] > 0.01:
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

def train_apriori(transactions_data, min_support=0.02, min_confidence=0.3):
    """Train Apriori model to find association rules"""
    global apriori_rules_df, apriori_products_df, is_apriori_trained

    try:
        basket_list = []
        product_info = {}

        for transaction in transactions_data:
            product_ids = []
            for item in transaction["transaksiItem"]:
                product_id = item["produkId"]
                product_ids.append(product_id)

                if product_id not in product_info:
                    produk_data = item["produk"]
                    product_info[product_id] = {
                        "nama": produk_data["nama"],
                        "harga": produk_data.get("harga", 0),
                        "gambar": produk_data.get("gambar", ""),
                        "stok": produk_data.get("stok", 0),
                        "status": produk_data.get("status", True),
                        "UMKM": produk_data.get("UMKM"),
                        "kategori": produk_data.get("kategori"),
                        "ProdukVarian": produk_data.get("ProdukVarian", [])
                    }

            if product_ids:
                basket_list.append(list(set(product_ids)))

        all_products = sorted({pid for basket in basket_list for pid in basket})
        encoded_rows = []
        for basket in basket_list:
            row = {pid: (pid in basket) for pid in all_products}
            encoded_rows.append(row)
        df_encoded = pd.DataFrame(encoded_rows)

        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

        products_list = []
        for product_id, info in product_info.items():
            products_list.append({
                "product_id": product_id,
                **info
            })
        
        apriori_rules_df = rules
        apriori_products_df = pd.DataFrame(products_list)
        is_apriori_trained = True

        logger.info(f"Apriori trained: {len(basket_list)} transactions, {len(all_products)} products, {len(rules)} rules")
        return True

    except Exception as e:
        logger.error(f"Error training Apriori model: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
def get_apriori_recommendations(cart_items, num_recommendations=5):
    """Get product recommendations based on Apriori association rules"""
    global apriori_rules_df, apriori_products_df, is_apriori_trained

    if not is_apriori_trained:
        return []

    try:
        cart_ids = set([item["id"] for item in cart_items])
        if not cart_ids:
            return []

        matched_rules = []
        for _, row in apriori_rules_df.iterrows():
            antecedents = set(row["antecedents"])
            if antecedents.issubset(cart_ids):
                for pid in row["consequents"]:
                    if pid not in cart_ids:
                        matched_rules.append({
                            "product_id": pid,
                            "confidence": float(row["confidence"]),
                            "lift": float(row["lift"])
                        })

        if not matched_rules:
            return []
        
        product_scores = {}
        for rule in matched_rules:
            pid = rule["product_id"]
            if pid not in product_scores or rule["confidence"] > product_scores[pid]["confidence"]:
                product_scores[pid] = rule

        unique_rules = list(product_scores.values())
        unique_rules.sort(key=lambda x: (x["lift"], x["confidence"]), reverse=True)
        top_rules = unique_rules[:num_recommendations]

        # Gabungkan dengan info produk
        recommendations = []
        for rule in top_rules:
            info = apriori_products_df[apriori_products_df["product_id"] == rule["product_id"]]
            if not info.empty:
                p = info.iloc[0]
                recommendations.append({
                    "id": str(p["product_id"]),
                    "nama": str(p.get("nama", "")),
                    "harga": int(p.get("harga", 0)),
                    "gambar": str(p.get("gambar", "")),
                    "stok": int(p.get("stok", 0)),
                    "status": bool(p.get("status", True)),
                    "UMKM": p.get("UMKM"),
                    "kategori": p.get("kategori"),
                    "ProdukVarian": p.get("ProdukVarian", []),
                    "score": float(rule["confidence"]),
                    "lift": float(rule["lift"]),
                    "confidence": float(rule["confidence"])
                })

        return recommendations

    except Exception as e:
        logger.error(f"Error getting Apriori recommendations: {str(e)}")
        return []



# def train_collaborative_filtering(transactions_data):
#     """Train collaborative filtering model"""
#     global collaborative_model, collaborative_products_df, is_collaborative_trained
    
#     try:
#         # Prepare transaction data for matrix factorization
#         ratings_list = []
#         product_info = {}
        
#         for transaction in transactions_data:
#             transaction_id = transaction['id']
            
#             for item in transaction['transaksiItem']:
#                 product_id = item['produkId']  # This should be produkId from TransaksiItem
#                 quantity = item['jumlah']
#                 rating = convert_quantity_to_rating(quantity)
                
#                 ratings_list.append({
#                     'user_id': transaction_id,
#                     'item_id': product_id,
#                     'rating': rating
#                 })
                
#                 # Store product info for later use
#                 if product_id not in product_info:
#                     product_info[product_id] = {
#                         'nama': item['produk']['nama']
#                     }
        
#         if not ratings_list:
#             logger.error("No transaction data provided for collaborative filtering")
#             return False
        
#         # Create DataFrame
#         ratings_df = pd.DataFrame(ratings_list)
        
#         # Create Surprise dataset
#         reader = Reader(rating_scale=(1, 5))
#         data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
        
#         # Train SVD model
#         collaborative_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
#         trainset = data.build_full_trainset()
#         collaborative_model.fit(trainset)
        
#         # Store product info
#         collaborative_products_df = pd.DataFrame.from_dict(product_info, orient='index')
#         collaborative_products_df.reset_index(inplace=True)
#         collaborative_products_df.rename(columns={'index': 'product_id'}, inplace=True)
        
#         is_collaborative_trained = True
        
#         logger.info(f"Collaborative model trained with {len(ratings_list)} ratings from {len(set([r['user_id'] for r in ratings_list]))} transactions")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error training collaborative model: {str(e)}")
#         return False

# def get_collaborative_recommendations(cart_items, all_products, num_recommendations=5):
#     """Get collaborative filtering recommendations"""
#     global collaborative_model, collaborative_products_df, is_collaborative_trained
    
#     if not is_collaborative_trained:
#         return []
    
#     try:
#         # Get cart product IDs
#         cart_product_ids = set([item['id'] for item in cart_items])
        
#         # Create a dummy user ID for prediction
#         dummy_user_id = "temp_user"
        
#         # Get all available products
#         all_product_ids = [p['id'] for p in all_products]
        
#         # Predict ratings for all products
#         predictions = []
#         for product_id in all_product_ids:
#             if product_id not in cart_product_ids:  # Don't recommend items already in cart
#                 pred = collaborative_model.predict(dummy_user_id, product_id)
#                 predictions.append((product_id, pred.est))
        
#         # Sort by predicted rating
#         predictions.sort(key=lambda x: x[1], reverse=True)
        
#         # Get top recommendations
#         top_predictions = predictions[:num_recommendations]
        
#         # Build recommendations with full product data
#         recommendations = []
#         for product_id, score in top_predictions:
#             # Find product in all_products
#             product_data = next((p for p in all_products if p['id'] == product_id), None)
#             if product_data:
#                 recommendation = {
#                     'id': product_data['id'],
#                     'nama': product_data['nama'],
#                     'harga': product_data['harga'],
#                     'gambar': product_data['gambar'],
#                     'stok': product_data['stok'],
#                     'status': product_data['status'],
#                     'score': float(score),
#                     'UMKM': product_data.get('UMKM'),
#                     'kategori': product_data.get('kategori'),
#                     'ProdukVarian': product_data.get('ProdukVarian', [])
#                 }
#                 recommendations.append(recommendation)
        
#         return recommendations
        
#     except Exception as e:
#         logger.error(f"Error getting collaborative recommendations: {str(e)}")
#         return []

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "hybrid-recommendation-system",
        "content_trained": is_content_trained,
        "apriori_trained": is_apriori_trained
    })

@app.route('/train-content', methods=['POST'])
def train_content_endpoint():
    """Train content-based model"""
    try:
        data = request.get_json()
        products_data = data.get('products', [])
        
        if not products_data:
            return  ({
                "success": False,
                "message": "No products data provided"
            }), 400
        
        success = train_content_based(products_data)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Content-based model trained successfully",
                "total_products": len(products_data),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to train content-based model"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in train-content endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# @app.route('/train-collaborative', methods=['POST'])
# def train_collaborative_endpoint():
#     """Train collaborative filtering model"""
#     try:
#         data = request.get_json()
#         transactions_data = data.get('transactions', [])
        
#         if not transactions_data:
#             return jsonify({"error": "No transactions data provided"}), 400
        
#         success = train_collaborative_filtering(transactions_data)
        
#         if success:
#             return jsonify({
#                 "message": "Collaborative filtering model trained successfully",
#                 "total_transactions": len(transactions_data),
#                 "timestamp": datetime.now().isoformat()
#             })
#         else:
#             return jsonify({"error": "Failed to train collaborative model"}), 500
            
#     except Exception as e:
#         logger.error(f"Error in train-collaborative endpoint: {str(e)}")
#         return jsonify({"error": str(e)}), 500

@app.route('/train-apriori', methods=['POST'])
def train_apriori_endpoint():
    """Train Apriori model with transaction data"""
    try:
        data = request.get_json()
        transactions_data = data.get('transactions', [])
        min_support = data.get('min_support', 0.02)
        min_confidence = data.get('min_confidence', 0.3)

        if not transactions_data:
            return jsonify({
                "success": False,
                "message": "No transactions data provided"
            }), 400

        success = train_apriori(transactions_data, min_support, min_confidence)

        if not success:
            return jsonify({
                "success": False,
                "message": "Failed to train Apriori model"
            }), 500

        return jsonify({
            "success": True,
            "message": "Apriori model trained successfully",
            "transactions_count": len(transactions_data),
            "min_support": min_support,
            "min_confidence": min_confidence
        })

    except Exception as e:
        logger.error(f"Error in train-apriori endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/recommend-content', methods=['POST'])
def recommend_content_endpoint():
    """Get content-based recommendations"""
    try:
        data = request.get_json()
        cart_items = data.get('items', [])
        num_recommendations = data.get('num_recommendations', 5)
        
        if not cart_items:
            return jsonify({
                "success": False,
                "message": "No cart items provided"
            }), 400
        
        recommendations = get_content_recommendations(cart_items, num_recommendations)
        
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "method": "content_based",
            "total_found": len(recommendations),
            "cart_items_count": len(cart_items)
        })
        
    except Exception as e:
        logger.error(f"Error in recommend-content endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# @app.route('/recommend-collaborative', methods=['POST'])
# def recommend_collaborative_endpoint():
#     """Get collaborative filtering recommendations"""
#     try:
#         data = request.get_json()
#         cart_items = data.get('items', [])
#         all_products = data.get('all_products', [])
#         num_recommendations = data.get('num_recommendations', 5)
        
#         if not cart_items:
#             return jsonify({"error": "No cart items provided"}), 400
        
#         if not all_products:
#             return jsonify({"error": "No products data provided"}), 400
        
#         recommendations = get_collaborative_recommendations(cart_items, all_products, num_recommendations)
        
#         return jsonify({
#             "recommendations": recommendations,
#             "method": "collaborative",
#             "total_found": len(recommendations),
#             "cart_items_count": len(cart_items)
#         })
        
#     except Exception as e:
#         logger.error(f"Error in recommend-collaborative endpoint: {str(e)}")
#         return jsonify({"error": str(e)}), 500
    
@app.route('/recommend-apriori', methods=['POST'])
def recommend_apriori_endpoint():
    """Get Apriori recommendations"""
    try:
        data = request.get_json()
        cart_items = data.get('items', [])
        num_recommendations = data.get('num_recommendations', 5)

        if not cart_items:
            return jsonify({
                "success": False,
                "message": "No cart items provided"
            }), 400

        recommendations = get_apriori_recommendations(cart_items, num_recommendations)

        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "method": "apriori",
            "total_found": len(recommendations),
            "cart_items_count": len(cart_items)
        })

    except Exception as e:
        logger.error(f"Error in recommend-apriori endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Hybrid Recommendation System API")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)