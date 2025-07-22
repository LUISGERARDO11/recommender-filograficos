from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)

class Category(db.Model):
    __tablename__ = 'categories'
    category_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    parent_id = db.Column(db.Integer, db.ForeignKey('categories.category_id'), nullable=True)
    active = db.Column(db.Boolean, nullable=False, default=True)
    imagen_url = db.Column(db.String(255), nullable=True)
    public_id = db.Column(db.String(255), nullable=True)
    color_fondo = db.Column(db.String(10), nullable=True)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class Product(db.Model):
    __tablename__ = 'products'
    product_id = db.Column(db.Integer, primary_key=True)
    collaborator_id = db.Column(db.Integer, nullable=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.category_id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    product_type = db.Column(db.Enum('Existencia', 'Personalizado'), nullable=False)
    on_promotion = db.Column(db.Boolean, default=False)
    average_rating = db.Column(db.DECIMAL(3, 2), default=0)
    total_reviews = db.Column(db.Integer, default=0)
    status = db.Column(db.Enum('active', 'inactive'), default='active')
    standard_delivery_days = db.Column(db.Integer, nullable=False, default=1)
    urgent_delivery_enabled = db.Column(db.Boolean, nullable=False, default=False)
    urgent_delivery_days = db.Column(db.Integer, nullable=True)
    urgent_delivery_cost = db.Column(db.DECIMAL(10, 2), nullable=True, default=0.00)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)
    category = db.relationship('Category', backref='products')

class ProductVariant(db.Model):
    __tablename__ = 'product_variants'
    variant_id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'), nullable=False)
    sku = db.Column(db.String(50), nullable=False, unique=True)
    production_cost = db.Column(db.DECIMAL(10, 2), nullable=False)
    profit_margin = db.Column(db.DECIMAL(5, 2), nullable=False)
    calculated_price = db.Column(db.DECIMAL(10, 2), nullable=False)
    stock = db.Column(db.Integer, nullable=False, default=0)
    stock_threshold = db.Column(db.Integer, nullable=False, default=10)
    last_stock_added_at = db.Column(db.DateTime, nullable=True)
    is_deleted = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class ProductImage(db.Model):
    __tablename__ = 'product_images'
    image_id = db.Column(db.Integer, primary_key=True)
    variant_id = db.Column(db.Integer, db.ForeignKey('product_variants.variant_id'), nullable=False)
    image_url = db.Column(db.String(255), nullable=False)
    public_id = db.Column(db.String(255), nullable=False)
    order = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class Order(db.Model):
    __tablename__ = 'orders'
    order_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    total = db.Column(db.DECIMAL(10, 2), nullable=False)
    total_urgent_cost = db.Column(db.DECIMAL(10, 2), nullable=False, default=0.00)
    order_status = db.Column(db.Enum('pending', 'processing', 'shipped', 'delivered', 'cancelled'), nullable=False, default='pending')
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class OrderDetail(db.Model):
    __tablename__ = 'order_details'
    order_detail_id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.order_id'), nullable=False)
    variant_id = db.Column(db.Integer, db.ForeignKey('product_variants.variant_id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.DECIMAL(10, 2), nullable=False)
    subtotal = db.Column(db.DECIMAL(10, 2), nullable=False)
    additional_cost = db.Column(db.DECIMAL(10, 2), nullable=False, default=0.00)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)