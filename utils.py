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
            # Calcular promedio de ratings ponderado por cantidad
            ratings = [float(d.average_rating) * d.quantity for d in details if d.average_rating]
            total_items = sum(d.quantity for d in details if d.average_rating)
            avg_rating = sum(ratings) / total_items if total_items > 0 else 0
        else:
            total_quantity = 0
            num_categories = 0
            avg_rating = 0

        # Manejo de valores atípicos (igual que en el entrenamiento)
        df = pd.DataFrame({
            'user_id': [user_id],
            'total_spent': [total_spent],
            'num_orders': [num_orders],
            'total_quantity': [total_quantity],
            'num_categories': [num_categories],
            'avg_rating': [avg_rating]
        })

        # Aplicar clip como en el entrenamiento
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
            Product.average_rating,
            Category.name.label('category_name'),
            func.min(ProductVariant.calculated_price).label('min_price'),
            func.max(ProductVariant.calculated_price).label('max_price'),
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
            'average_rating': float(p.average_rating) if p.average_rating else 0,
            'category': p.category_name,
            'min_price': float(p.min_price) if p.min_price else None,
            'max_price': float(p.max_price) if p.max_price else None,
            'image_url': p.image_url
        } for p in products]
        
    except Exception as e:
        print(f"Error in get_product_details: {str(e)}")
        return []

def get_recommendations(
    user_id: int,
    purchased_products: Optional[List[str]],
    scaler,
    kmeans,
    rules: pd.DataFrame,
    db_session,
    user_data: Optional[Dict] = None
) -> Tuple[Optional[int], List[Dict[str, Union[str, float]]]]:
    """
    Generate product recommendations with detailed product information.
    
    Args:
        user_id (int): ID of the user
        purchased_products (List[str]): List of product names purchased (optional)
        scaler: Pre-trained StandardScaler model
        kmeans: Pre-trained KMeans model
        rules (pd.DataFrame): DataFrame with association rules
        db_session: SQLAlchemy session
        user_data (Dict, optional): Pre-computed user features for new users
    
    Returns:
        Tuple[Optional[int], List[Dict]]: Cluster ID and list of recommendations with product details
    """
    try:
        if user_data:
            # Para nuevos usuarios sin historial
            df = pd.DataFrame([user_data])
        else:
            # Para usuarios existentes
            user_features = get_user_features(user_id, db_session)
            if user_features is None or user_features.empty:
                # Valores por defecto para usuarios sin historial
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

        # Verificar que tenemos todas las características necesarias
        required_features = ['total_spent', 'num_orders', 'total_quantity', 'num_categories', 'avg_rating']
        if not all(feat in df.columns for feat in required_features):
            missing = [feat for feat in required_features if feat not in df.columns]
            print(f"Missing features in user data: {missing}")
            df[missing] = 0  # Asignar valores por defecto

        # Escalar características y predecir cluster
        X_scaled = scaler.transform(df[required_features])
        cluster = kmeans.predict(X_scaled)[0]

        # Filtrar reglas si hay productos comprados
        if purchased_products:
            filtered_rules = rules[rules['antecedents'].apply(
                lambda x: any(p in x for p in purchased_products)
            )]
            # Si no encontramos reglas, usar las generales
            if filtered_rules.empty:
                filtered_rules = rules
        else:
            filtered_rules = rules

        # Obtener las mejores recomendaciones
        top_rules = filtered_rules.sort_values(
            by=['confidence', 'lift'], 
            ascending=False
        ).head(5)

        # Procesar recomendaciones
        recommendations = []
        product_names = []
        
        for _, rule in top_rules.iterrows():
            try:
                for product_name in list(rule['consequents']):
                    if product_name not in product_names:
                        product_names.append(product_name)
                        recommendations.append({
                            'product_name': product_name,
                            'confidence': float(rule['confidence']),
                            'lift': float(rule['lift'])
                        })
                        if len(recommendations) >= 5:
                            break
            except Exception as e:
                print(f"Error formatting recommendation: {str(e)}")
                continue
            if len(recommendations) >= 5:
                break

        # Obtener detalles completos de los productos
        product_details = get_product_details(product_names, db_session)

        # Combinar métricas de recomendación con detalles de producto
        for rec in recommendations:
            product_detail = next(
                (p for p in product_details if p.get('name') == rec['product_name']), 
                {}
            )
            rec.update(product_detail)

        return cluster, recommendations

    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return None, []