"""
Authentication utilities for Flask backend
Handles JWT tokens, password hashing, and authentication decorators
"""

import os
from functools import wraps
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask import request, jsonify
from flask_jwt_extended import (
    JWTManager, create_access_token, get_jwt_identity, 
    verify_jwt_in_request, get_jwt
)
from database_utils import db_manager

# JWT Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-secret-key-change-in-production')
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', '24')))


def init_jwt(app):
    """Initialize JWT manager with Flask app"""
    app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = JWT_ACCESS_TOKEN_EXPIRES
    jwt = JWTManager(app)
    return jwt


def hash_password(password: str) -> str:
    """Hash a password using werkzeug's secure method"""
    return generate_password_hash(password, method='pbkdf2:sha256')


def verify_password(password_hash: str, password: str) -> bool:
    """Verify a password against its hash"""
    return check_password_hash(password_hash, password)


def create_token(username: str, role: str) -> str:
    """Create a JWT token for a user"""
    additional_claims = {
        "role": role
    }
    return create_access_token(
        identity=username, 
        additional_claims=additional_claims,
        expires_delta=JWT_ACCESS_TOKEN_EXPIRES
    )


def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": "Invalid or missing token", "details": str(e)}), 401
    return decorated


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            verify_jwt_in_request()
            claims = get_jwt()
            if claims.get('role') != 'admin':
                return jsonify({"error": "Admin access required"}), 403
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": "Invalid or missing token", "details": str(e)}), 401
    return decorated


def user_allowed_required(f):
    """Decorator to check if user is not blocked"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            verify_jwt_in_request()
            username = get_jwt_identity()
            
            # Check if user is allowed
            if not db_manager.is_user_allowed(username):
                return jsonify({
                    "error": "Account blocked",
                    "message": "Your account has been blocked. Please contact support."
                }), 403
            
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": "Authentication failed", "details": str(e)}), 401
    return decorated


def authenticate_user(username: str, password: str) -> dict:
    """
    Authenticate a user with username and password
    Returns dict with success status, token, and user info
    """
    user = db_manager.get_auth_user(username)
    
    if not user:
        return {
            "success": False,
            "message": "Invalid username or password"
        }
    
    # Check if user is blocked
    if not user.get('allowed', False):
        return {
            "success": False,
            "message": "Account blocked. Please contact support.",
            "unauthorized": True
        }
    
    # Verify password
    if not verify_password(user['password_hash'], password):
        return {
            "success": False,
            "message": "Invalid username or password"
        }
    
    # Update last login
    db_manager.update_last_login(username)
    
    # Create token
    token = create_token(username, user['role'])
    
    return {
        "success": True,
        "token": token,
        "username": username,
        "role": user['role'],
        "display_name": user.get('display_name', username)
    }


def register_user(username: str, password: str, display_name: str = None) -> dict:
    """
    Register a new user
    Returns dict with success status and message
    """
    # Validate username
    if not username or len(username) < 3:
        return {
            "success": False,
            "message": "Username must be at least 3 characters"
        }
    
    # Validate password
    if not password or len(password) < 6:
        return {
            "success": False,
            "message": "Password must be at least 6 characters"
        }
    
    # Hash password
    password_hash = hash_password(password)
    
    # Create user
    success = db_manager.create_auth_user(
        username=username,
        password_hash=password_hash,
        role='user',
        display_name=display_name or username
    )
    
    if not success:
        return {
            "success": False,
            "message": "Username already exists"
        }
    
    # Create token for auto-login
    token = create_token(username, 'user')
    
    return {
        "success": True,
        "message": "Registration successful",
        "token": token,
        "username": username,
        "role": "user"
    }

