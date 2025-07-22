import pandas as pd
from sqlalchemy import func
from typing import List, Dict, Tuple, Optional, Union
from models import Order, OrderDetail, Product, ProductVariant, ProductImage, Category

def get_user_features(user_id: int, db_session) -> Optional[pd.DataFrame]:
    """
    Compute user features from the database for clustering, aligned with training preprocessing.
    
    Args:
        user_id (int): ID of the user
        db_session: SQLAlchemy session
    
    Returns:
        pandas.DataFrame: DataFrame with user features:
            - total_spent: Total user spending
            - num_orders: Number of orders
            - total_quantity: Total quantity of products purchased
            - num_categories: Number of unique categories purchased
            - avg_rating: Average rating of purchased products
    """
    try:
        # Obtener todas las órdenes del usuario
        orders = db_session.query(Order).filter_by(user_id=user_id).all()
        
        if not orders:
            return pd.DataFrame({
                'user_id': [user_id],
                'total_spent': [0],
                'num_orders': [0],
                'total_quantity': [0],
                'num_categories': [0],
                'avg_rating': [0]
            })

        # Obtener IDs de órdenes para los detalles
        order_ids = [o.order_id for o in orders]
        
        # Obtener detalles de las órdenes con información de categoría y rating
        details = db_session.query(
            OrderDetail.order_id,
            OrderDetail.quantity,
            OrderDetail.subtotal,
            Product.category_id,
            Product.average_rating
        ).join(
            ProductVariant, OrderDetail.variant_id == ProductVariant.variant_id
        ).join(
            Product, ProductVariant.product_id == Product.product_id
        ).filter(
            OrderDetail.order_id.in_(order_ids)
        ).all()

        # Calcular métricas
        total_spent = sum(float(o.total) for o in orders)
        num_orders = len(orders)
        
        if details:
            total_quantity = sum(d.quantity for d in details)
            num_categories = len(set(d.category_id for d in details))
            # Promedio simple de ratings, como en la libreta
            ratings = [float(d.average_rating) for d in details if d.average_rating]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
        else:
            total_quantity = 0
            num_categories = 0
            avg_rating = 0

        # Crear DataFrame
        df = pd.DataFrame({
            'user_id': [user_id],
            'total_spent': [total_spent],
            'num_orders': [num_orders],
            'total_quantity': [total_quantity],
            'num_categories': [num_categories],
            'avg_rating': [avg_rating]
        })

        # Aplicar clip como en la libreta
        df['total_spent'] = df['total_spent'].clip(upper=df['total_spent'].quantile(0.95) if not df['total_spent'].empty else total_spent)
        df['total_quantity'] = df['total_quantity'].clip(upper=df['total_quantity'].quantile(0.95) if not df['total_quantity'].empty else total_quantity)

        return df

    except Exception as e:
        print(f"Error in get_user_features: {str(e)}")
        return None

def get_product_details(product_names: List[str], db_session) -> List[Dict]:
    """
    Fetch detailed product information for a list of product names.
    
    Args:
        product_names (List[str]): List of product names
        db_session: SQLAlchemy session
    
    Returns:
        List[Dict]: List of product details
    """
    try:
        products = db_session.query(
            Product.product_id,
            Product.name,
            Product.description,
            Product.product_type,
            Product.average_rating,
            Product.total_reviews,
            Product.standard_delivery_days,
            Product.urgent_delivery_enabled,
            Product.urgent_delivery_days,
            Product.urgent_delivery_cost,
            Product.created_at,
            Product.updated_at,
            Category.name.label('category_name'),
            func.min(ProductVariant.calculated_price).label('min_price'),
            func.max(ProductVariant.calculated_price).label('max_price'),
            func.sum(ProductVariant.stock).label('total_stock'),
            func.count(ProductVariant.variant_id).label('variant_count'),
            db_session.query(ProductImage.image_url)
                .join(ProductVariant, ProductImage.variant_id == ProductVariant.variant_id)
                .filter(ProductVariant.product_id == Product.product_id)
                .filter(ProductImage.order == 1)
                .filter(ProductVariant.is_deleted == False)
                .limit(1)
                .correlate(Product)
                .as_scalar()
                .label('image_url')
        ).join(
            Category, Product.category_id == Category.category_id, isouter=True
        ).join(
            ProductVariant, Product.product_id == ProductVariant.product_id, isouter=True
        ).filter(
            Product.name.in_(product_names),
            Product.status == 'active',
            ProductVariant.is_deleted == False
        ).group_by(
            Product.product_id
        ).all()
        
        return [{
            'product_id': p.product_id,
            'name': p.name,
            'description': p.description,
            'product_type': p.product_type,
            'average_rating': float(p.average_rating) if p.average_rating else 0,
            'total_reviews': int(p.total_reviews) if p.total_reviews else 0,
            'min_price': float(p.min_price) if p.min_price else None,
            'max_price': float(p.max_price) if p.max_price else None,
            'total_stock': int(p.total_stock) if p.total_stock else 0,
            'variant_count': int(p.variant_count) if p.variant_count else 0,
            'category': p.category_name,
            'image_url': p.image_url,
            'created_at': p.created_at.isoformat() if p.created_at else None,
            'updated_at': p.updated_at.isoformat() if p.updated_at else None,
            'collaborator': None,
            'standard_delivery_days': int(p.standard_delivery_days) if p.standard_delivery_days else None,
            'urgent_delivery_enabled': bool(p.urgent_delivery_enabled),
            'urgent_delivery_days': int(p.urgent_delivery_days) if p.urgent_delivery_days else None,
            'urgent_delivery_cost': float(p.urgent_delivery_cost) if p.urgent_delivery_cost else None
        } for p in products]
        
    except Exception as e:
        print(f"Error in get_product_details: {str(e)}")
        return []

def get_recommendations(
    user_id: int,
    purchased_products: Optional[List[str]],
    scaler,
    kmeans,
    rules_by_cluster: Dict[int, pd.DataFrame],
    db_session,
    user_data: Optional[Dict] = None
) -> Tuple[Optional[int], List[Dict[str, Union[str, float]]]]:
    """
    Generate product recommendations with detailed product information based on cluster-specific rules.
    
    Args:
        user_id (int): ID of the user
        purchased_products (List[str]): List of product names purchased (optional)
        scaler: Pre-trained StandardScaler model
        kmeans: Pre-trained KMeans model
        rules_by_cluster (Dict[int, pd.DataFrame]): Dictionary of association rules by cluster
        db_session: SQLAlchemy session
        user_data (Dict, optional): Pre-computed user features for new users
    
    Returns:
        Tuple[Optional[int], List[Dict]]: Cluster ID and list of recommendations with product details
    """
    try:
        # Obtener características del usuario
        if user_data:
            df = pd.DataFrame([user_data])
        else:
            user_features = get_user_features(user_id, db_session)
            if user_features is None or user_features.empty:
                df = pd.DataFrame({
                    'user_id': [user_id],
                    'total_spent': [0],
                    'num_orders': [0],
                    'total_quantity': [0],
                    'num_categories': [0],
                    'avg_rating': [0]
                })
            else:
                df = user_features

        # Verificar características requeridas
        required_features = ['total_spent', 'num_orders', 'total_quantity', 'num_categories', 'avg_rating']
        missing_features = [feat for feat in required_features if feat not in df.columns]
        if missing_features:
            df[missing_features] = 0

        # Escalar características y predecir cluster
        X_scaled = scaler.transform(df[required_features])
        cluster = kmeans.predict(X_scaled)[0]

        # Obtener productos comprados desde la base de datos si no se proporcionan
        if not purchased_products:
            user_products = db_session.query(Product.name).join(
                ProductVariant, Product.product_id == ProductVariant.product_id
            ).join(
                OrderDetail, ProductVariant.variant_id == OrderDetail.variant_id
            ).join(
                Order, OrderDetail.order_id == Order.order_id
            ).filter(
                Order.user_id == user_id
            ).distinct().all()
            purchased_products = [p[0] for p in user_products if p[0]]

        # Obtener reglas del cluster
        cluster_rules = rules_by_cluster.get(cluster, pd.DataFrame())
        if cluster_rules.empty:
            return cluster, []

        # Filtrar reglas basadas en productos comprados
        if purchased_products:
            valid_products = db_session.query(Product.name).filter(Product.name.in_(purchased_products), Product.status == 'active').all()
            valid_products = [p[0] for p in valid_products]
            filtered_rules = cluster_rules[cluster_rules['antecedents'].apply(lambda x: any(p in x for p in valid_products))]
            if filtered_rules.empty:
                filtered_rules = cluster_rules  # Usar todas las reglas del cluster si no hay coincidencias
        else:
            filtered_rules = cluster_rules

        # Obtener las mejores recomendaciones
        recommendations = []
        seen_products = set(purchased_products)
        for _, rule in filtered_rules.sort_values(by=['confidence', 'lift'], ascending=False).iterrows():
            for product_name in list(rule['consequents']):
                if product_name not in seen_products:
                    seen_products.add(product_name)
                    recommendations.append({
                        'product_name': product_name,
                        'confidence': float(rule['confidence']),
                        'lift': float(rule['lift'])
                    })
                    if len(recommendations) >= 5:
                        break
            if len(recommendations) >= 5:
                break

        # Obtener detalles completos de los productos
        product_names = [r['product_name'] for r in recommendations]
        product_details = get_product_details(product_names, db_session)

        # Combinar métricas de recomendación con detalles de producto
        for rec in recommendations:
            product_detail = next((p for p in product_details if p.get('name') == rec['product_name']), {})
            rec.update(product_detail)
            rec.pop('product_name', None)  # Eliminar product_name para evitar duplicación con name

        return cluster, recommendations

    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return None, []

def get_cluster_summary(db_session, scaler, kmeans) -> Dict:
    """
    Fetch cluster summary based on the original clustering analysis.
    
    Args:
        db_session: SQLAlchemy session
        scaler: Pre-trained StandardScaler model
        kmeans: Pre-trained KMeans model
    
    Returns:
        Dict: Cluster summary with fields expected by the API
    """
    try:
        # Obtener datos de usuarios desde la base de datos
        users_data = db_session.query(
            Order.user_id,
            func.sum(Order.total).label('total_spent'),
            func.count(Order.order_id).label('num_orders'),
            func.sum(OrderDetail.quantity).label('total_quantity'),
            func.count(func.distinct(Product.category_id)).label('num_categories'),
            func.avg(Product.average_rating).label('avg_rating')
        ).join(
            OrderDetail, Order.order_id == OrderDetail.order_id
        ).join(
            ProductVariant, OrderDetail.variant_id == ProductVariant.variant_id
        ).join(
            Product, ProductVariant.product_id == Product.product_id
        ).group_by(
            Order.user_id
        ).all()

        if not users_data:
            return {}

        # Crear DataFrame
        df = pd.DataFrame(
            [(u.user_id, float(u.total_spent), u.num_orders, u.total_quantity, u.num_categories, float(u.avg_rating or 0))
             for u in users_data],
            columns=['user_id', 'total_spent', 'num_orders', 'total_quantity', 'num_categories', 'avg_rating']
        )

        # Aplicar clip como en la libreta
        df['total_spent'] = df['total_spent'].clip(upper=df['total_spent'].quantile(0.95))
        df['total_quantity'] = df['total_quantity'].clip(upper=df['total_quantity'].quantile(0.95))

        # Escalar características
        required_features = ['total_spent', 'num_orders', 'total_quantity', 'num_categories', 'avg_rating']
        X_scaled = scaler.transform(df[required_features])

        # Predecir clusters
        df['cluster'] = kmeans.predict(X_scaled)

        # Generar resumen
        cluster_summary = df.groupby('cluster').agg({
            'total_spent': 'mean',
            'num_orders': 'mean',
            'total_quantity': 'mean',
            'num_categories': 'mean',
            'avg_rating': 'mean',
            'user_id': 'count'
        }).reset_index()
        cluster_summary.columns = [
            'cluster', 'total_spent', 'number_of_orders', 'total_units',
            'number_of_categories', 'average_rating', 'number_of_users'
        ]
        cluster_summary['average_order_quantity'] = cluster_summary['total_units'] / cluster_summary['number_of_orders']

        # Convertir a formato esperado por la API
        return {
            int(row['cluster']): {
                'average_order_quantity': float(row['average_order_quantity']),
                'total_spent': float(row['total_spent']),
                'number_of_orders': int(row['number_of_orders']),
                'total_units': int(row['total_units']),
                'number_of_users': int(row['number_of_users'])
            } for _, row in cluster_summary.iterrows()
        }

    except Exception as e:
        print(f"Error in get_cluster_summary: {str(e)}")
        return {}