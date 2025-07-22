from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import pandas as pd
import pickle
from config import Config
from models import db, User, Product, Order, OrderDetail
from utils import get_user_features, get_product_details, get_recommendations

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
cache = Cache(app)

# Cargar los modelos
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/rules.pkl', 'rb') as f:
    rules = pickle.load(f)
with open('models/transaction_encoder.pkl', 'rb') as f:
    te = pickle.load(f)

@app.route('/recommendations', methods=['GET'])
@cache.cached(timeout=3600, key_prefix=lambda: f'recommend_{request.args.get("user_id")}_{pd.Timestamp.now().strftime("%Y%m%d%H")}')
def recommend():
    try:
        user_id = request.args.get('user_id', type=int)
        if not user_id:
            return jsonify({'error': 'user_id parameter is required'}), 400

        with db.session() as session:
            # Obtener características del usuario
            user_features = get_user_features(user_id, session)
            if user_features is None:
                return jsonify({'error': 'Error processing user data'}), 500

            # Definir orden exacto de características como en el entrenamiento
            features_order = ['total_spent', 'num_orders', 'total_quantity', 'num_categories', 'avg_rating']

            # Verificar que tenemos todas las características necesarias
            missing_features = [f for f in features_order if f not in user_features.columns]
            if missing_features:
                user_features[missing_features] = 0  # Valores por defecto

            # Ordenar columnas exactamente como durante el entrenamiento
            user_features = user_features[features_order]

            # Escalar características
            scaled_features = scaler.transform(user_features)

            # Predecir cluster
            cluster = kmeans.predict(scaled_features)[0]

            # Obtener recomendaciones generales (sin productos comprados)
            recommendations = []
            seen_products = set()

            for _, rule in rules.sort_values(by='confidence', ascending=False).iterrows():
                for product in list(rule['consequents']):
                    if product not in seen_products:
                        seen_products.add(product)
                        recommendations.append({
                            'product_name': product,
                            'confidence': float(rule['confidence']),
                            'lift': float(rule['lift'])
                        })
                        if len(recommendations) >= 5:
                            break
                if len(recommendations) >= 5:
                    break

            # Obtener detalles de productos
            product_names = [r['product_name'] for r in recommendations]
            product_details = get_product_details(product_names, session)

            # Combinar datos
            for rec, detail in zip(recommendations, product_details):
                rec.update(detail)

            return jsonify({
                'user_id': user_id,
                'cluster': int(cluster),
                'recommendations': recommendations
            })

    except Exception as e:
        app.logger.error(f"Error processing recommendation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend_with_products():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        purchased_products = data.get('purchased_products', [])
        user_data = data.get('user_data')  # Para nuevos usuarios

        if not user_id:
            return jsonify({'error': 'user_id is required', 'status': 'Invalid request'}), 400

        with db.session() as session:
            user = session.query(User).filter_by(user_id=user_id).first()
            status = 'User registered'
            if not user and not user_data:
                return jsonify({'error': 'User not found and no user_data provided', 'status': 'Invalid request'}), 400

            # Validar productos comprados
            if purchased_products:
                valid_products = session.query(Product.name).filter(Product.name.in_(purchased_products), Product.status == 'active').all()
                valid_products = [p[0] for p in valid_products]
                purchased_products = [p for p in purchased_products if p in valid_products]
                if not purchased_products and not user_data:
                    cluster, recs = get_recommendations(user_id, [], scaler, kmeans, rules, session)
                    return jsonify({
                        'user_id': user_id,
                        'cluster': int(cluster),
                        'recommendations': recs,
                        'status': 'No valid purchased products, using default recommendations'
                    })

            # Generar recomendaciones
            cluster, recommendations = get_recommendations(user_id, purchased_products, scaler, kmeans, rules, session, user_data)
            if not recommendations:
                return jsonify({
                    'error': f'No recommendations available for cluster {cluster}',
                    'status': 'No suitable rules found, using default recommendations'
                }), 404

            return jsonify({
                'user_id': user_id,
                'cluster': int(cluster),
                'recommendations': recommendations,
                'status': f'Recommendations based on {status.lower()}'
            })

    except Exception as e:
        app.logger.error(f"Error processing recommendation with products for user {user_id}: {e}")
        return jsonify({'error': str(e), 'status': 'Error occurred'}), 500

@app.route('/cluster_info', methods=['GET'])
def get_cluster_info():
    try:
        # Información de clústeres basada en el análisis de la libreta
        cluster_info = {
            0: {"description": "Low spending, few orders", "avg_spent": 150, "avg_orders": 2},
            1: {"description": "Medium spending, regular orders", "avg_spent": 450, "avg_orders": 5},
            2: {"description": "High spending, frequent orders", "avg_spent": 900, "avg_orders": 8},
            3: {"description": "Very high spending, VIP customers", "avg_spent": 2000, "avg_orders": 12}
        }
        return jsonify(cluster_info)
    except Exception as e:
        app.logger.error(f"Error fetching cluster info: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)