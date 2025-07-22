import pandas as pd
import joblib
from models import Order, OrderDetail, Product, ProductVariant, ProductImage, Category
from sqlalchemy import func
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

def get_user_features(user_id: int, db_session) -> Optional[pd.DataFrame]:
    """
    Compute user features from the database for clustering.
    
    Args:
        user_id (int): ID of the user.
        db_session: SQLAlchemy session for database queries.
    
    Returns:
        pandas.DataFrame: User features or default DataFrame if no orders found.
    """
    try:
        orders = db_session.query(Order).filter_by(user_id=user_id).all()
        if not orders:
            return pd.DataFrame({
                'user_id': [user_id],
                'gasto_total': [0],
                'numero_pedidos': [0],
                'unidades_totales': [0],
                'cantidad_promedio_pedido': [0]
            })

        df_orders = pd.DataFrame(
            [(o.user_id, o.total + o.total_urgent_cost, 1) for o in orders],
            columns=['user_id', 'subtotal', 'order_count']
        )

        order_ids = [o.order_id for o in orders]
        details = db_session.query(
            OrderDetail.order_id,
            OrderDetail.quantity
        ).filter(OrderDetail.order_id.in_(order_ids)).all()

        if not details:
            df_details = pd.DataFrame(columns=['order_id', 'order_quantity'])
        else:
            df_details = pd.DataFrame(
                [(d.order_id, d.quantity) for d in details],
                columns=['order_id', 'order_quantity']
            )

        user_features = df_orders.groupby('user_id').agg({
            'subtotal': 'sum',
            'order_count': 'sum'
        }).reset_index()

        user_features.columns = ['user_id', 'gasto_total', 'numero_pedidos']

        if not df_details.empty:
            user_features['unidades_totales'] = df_details['order_quantity'].sum()
            avg_per_order = df_details.groupby('order_id')['order_quantity'].sum().mean()
            user_features['cantidad_promedio_pedido'] = avg_per_order
        else:
            user_features['unidades_totales'] = 0
            user_features['cantidad_promedio_pedido'] = 0

        return user_features.fillna(0)

    except Exception as e:
        print(f"Error fetching user features: {str(e)}")
        return None

def get_product_details(product_names: List[str], db_session) -> List[Dict]:
    """
    Fetch detailed product information for a list of product names, similar to Node.js getHomeData format.
    
    Args:
        product_names (List[str]): List of product names to query.
        db_session: SQLAlchemy session for database queries.
    
    Returns:
        List[Dict]: List of product details in the format expected by the frontend.
    """
    try:
        products = db_session.query(
            Product.product_id,
            Product.name,
            Product.description,
            Product.product_type,
            Product.average_rating,
            Product.total_reviews,
            Product.created_at,
            Product.updated_at,
            Product.collaborator_id,
            Product.standard_delivery_days,
            Product.urgent_delivery_enabled,
            Product.urgent_delivery_days,
            Product.urgent_delivery_cost,
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

        formatted_products = []
        for product in products:
            formatted_products.append({
                'product_id': product.product_id,
                'name': product.name,
                'description': product.description,
                'product_type': product.product_type,
                'average_rating': float(product.average_rating) if product.average_rating else '0.00',
                'total_reviews': product.total_reviews or 0,
                'min_price': float(product.min_price) if product.min_price else None,
                'max_price': float(product.max_price) if product.max_price else None,
                'total_stock': int(product.total_stock) if product.total_stock else 0,
                'variant_count': int(product.variant_count) if product.variant_count else 0,
                'category': product.category_name,
                'image_url': product.image_url,
                'created_at': product.created_at.isoformat() if product.created_at else None,
                'updated_at': product.updated_at.isoformat() if product.updated_at else None,
                'collaborator': f"Collaborator {product.collaborator_id}" if product.collaborator_id else None,
                'standard_delivery_days': product.standard_delivery_days,
                'urgent_delivery_enabled': product.urgent_delivery_enabled,
                'urgent_delivery_days': product.urgent_delivery_days,
                'urgent_delivery_cost': float(product.urgent_delivery_cost) if product.urgent_delivery_cost else None
            })

        # Preserve the order of product_names
        product_dict = {p['name']: p for p in formatted_products}
        return [product_dict.get(name, {}) for name in product_names if name in product_dict]

    except Exception as e:
        print(f"Error fetching product details: {str(e)}")
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
    Generate product recommendations with detailed product information.
    
    Args:
        user_id (int): ID of the user.
        purchased_products (List[str]): List of product names purchased (optional).
        scaler: Pre-trained StandardScaler model.
        kmeans: Pre-trained KMeans model.
        rules_by_cluster (Dict[int, pd.DataFrame]): Dictionary of association rules by cluster.
        db_session: SQLAlchemy session for database queries.
        user_data (Dict, optional): Pre-computed user features for new users.
    
    Returns:
        Tuple[Optional[int], List[Dict]]: Cluster ID and list of recommendations with product details.
    """
    try:
        if user_data:
            df = pd.DataFrame([user_data])
        else:
            user_features = get_user_features(user_id, db_session)
            if user_features is None or user_features.empty:
                df = pd.DataFrame({
                    'user_id': [user_id],
                    'gasto_total': [0],
                    'numero_pedidos': [0],
                    'unidades_totales': [0],
                    'cantidad_promedio_pedido': [0]
                })
            else:
                df = user_features

        caracteristicas = ['cantidad_promedio_pedido', 'gasto_total', 'numero_pedidos', 'unidades_totales']
        if not all(feat in df.columns for feat in caracteristicas):
            missing = [feat for feat in caracteristicas if feat not in df.columns]
            print(f"Missing features in user data: {missing}")
            df[missing] = 0

        X_scaled = scaler.transform(df[caracteristicas])
        cluster = kmeans.predict(X_scaled)[0]

        rules = rules_by_cluster.get(cluster, rules_by_cluster.get(0, pd.DataFrame()))
        if rules.empty:
            return cluster, []

        if purchased_products:
            rules = rules[rules['antecedents'].apply(
                lambda x: any(p in str(x) for p in purchased_products)
            )]

        if rules.empty and purchased_products:
            rules = rules_by_cluster.get(0, pd.DataFrame())

        if rules.empty:
            return cluster, []

        recommendations = rules[['consequents', 'confidence', 'lift']].head(5)
        recommendations = recommendations.to_dict(orient='records')

        formatted_recommendations = []
        product_names = []
        for rule in recommendations:
            try:
                if rule['consequents']:
                    product_name = list(rule['consequents'])[0]
                    product_names.append(product_name)
                    formatted_recommendations.append({
                        'product_name': product_name,
                        'confidence': float(rule['confidence']),
                        'lift': float(rule['lift'])
                    })
            except Exception as e:
                print(f"Error formatting recommendation: {str(e)}")
                continue

        # Fetch detailed product information
        product_details = get_product_details(product_names, db_session)

        # Merge recommendation metrics with product details
        for rec in formatted_recommendations:
            product_detail = next((p for p in product_details if p.get('name') == rec['product_name']), {})
            rec.update(product_detail)

        return cluster, formatted_recommendations

    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return None, []
        