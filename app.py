from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import pandas as pd
import joblib
from config import Config
from models import db, User, Product, Order, OrderDetail, ProductVariant
from utils import get_user_features, get_recommendations

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
cache = Cache(app)

# Load pre-trained models and rules globally
scaler = joblib.load('data/scaler.pkl')
kmeans = joblib.load('data/kmeans_model.pkl')
rules_by_cluster = {i: joblib.load(f'data/rules_cluster_{i}.pkl') for i in range(4)}

@app.route('/recommendations', methods=['GET'])
@cache.cached(timeout=3600, key_prefix=lambda: f'recommend_{request.args.get("user_id")}_{pd.Timestamp.now().strftime("%Y%m%d%H")}')
def recommend():
    try:
        user_id = request.args.get('user_id', type=int)
        if not user_id:
            return jsonify({'error': 'user_id parameter is required', 'status': 'Invalid request'}), 400

        with db.session() as session:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                return jsonify({'error': 'User not found', 'status': 'User not registered'}), 404
            
            cluster, recommendations = get_recommendations(user_id, [], scaler, kmeans, rules_by_cluster, session)
            if not recommendations:
                return jsonify({'error': f'No recommendations available for cluster {cluster}', 'status': 'No purchase history, using default cluster recommendations'}), 404
            
            return jsonify({
                'user_id': user_id,
                'cluster': int(cluster),
                'recommendations': recommendations,
                'status': 'Recommendations based on cluster and purchase history'
            })
    except Exception as e:
        app.logger.error(f"Error processing recommendation for user {user_id}: {e}")
        return jsonify({'error': str(e), 'status': 'Error occurred'}), 500

@app.route('/recommend', methods=['POST'])
def recommend_with_products():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        purchased_products = data.get('purchased_products', [])
        user_data = data.get('user_data')  # For new users
        
        if not user_id:
            return jsonify({'error': 'user_id is required', 'status': 'Invalid request'}), 400

        with db.session() as session:
            user = session.query(User).filter_by(user_id=user_id).first()
            status = 'User registered'
            if not user:
                status = 'New user, cluster predicted with provided data' if user_data else 'New user, using default cluster'
            
            if purchased_products:
                valid_products = session.query(Product.name).filter(Product.name.in_(purchased_products)).all()
                valid_products = [p[0] for p in valid_products]
                purchased_products = [p for p in purchased_products if p in valid_products]
                if not purchased_products and not user_data:
                    cluster, recs = get_recommendations(user_id, [], scaler, kmeans, rules_by_cluster, session)
                    return jsonify({
                        'user_id': user_id,
                        'cluster': int(cluster),
                        'recommendations': recs,
                        'status': 'No valid purchased products, using default cluster recommendations'
                    })

            cluster, recommendations = get_recommendations(user_id, purchased_products, scaler, kmeans, rules_by_cluster, session, user_data)
            if not recommendations:
                return jsonify({'error': f'No recommendations available for cluster {cluster}', 'status': 'No suitable rules found, using default recommendations'}), 404
            
            return jsonify({
                'user_id': user_id,
                'cluster': int(cluster),
                'recommendations': recommendations,
                'status': f'Recommendations based on {status.lower()}'
            })
    except Exception as e:
        app.logger.error(f"Error processing recommendation with products for user {user_id}: {e}")
        return jsonify({'error': str(e), 'status': 'Error occurred'}), 500

@app.route('/clusters', methods=['GET'])
def get_clusters():
    try:
        clusters = pd.read_csv('data/usuarios_clusterizados.csv')
        summary = clusters.groupby('cluster').agg({
            'cantidad_promedio_pedido': 'mean',
            'gasto_total': 'mean',
            'numero_pedidos': 'mean',
            'unidades_totales': 'mean',
            'user_id': 'count'
        }).rename(columns={'user_id': 'numero_usuarios'}).to_dict(orient='index')
        return jsonify(summary)
    except Exception as e:
        app.logger.error(f"Error fetching clusters: {e}")
        return jsonify({'error': str(e), 'status': 'Error fetching cluster summary'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)