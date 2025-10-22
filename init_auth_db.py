"""
Database initialization script for authentication system
Creates auth tables and default admin user
"""

import os
import sys
from getpass import getpass
from dotenv import load_dotenv

# Load environment variables from .env file FIRST (before importing database_utils)
load_dotenv()

# Now import modules that need environment variables
from database_utils import db_manager
from auth_backend_utils import hash_password


def init_auth_database():
    """Initialize authentication database"""
    print("=" * 60)
    print("Authentication Database Initialization")
    print("=" * 60)
    
    if not db_manager.connection:
        print("\n[!] Error: Database connection not available")
        print("    Make sure DATABASE_URL environment variable is set")
        return False
    
    print("\n[1] Creating authentication tables...")
    # Tables are created automatically by DatabaseManager.create_tables()
    print("    ✓ Tables created/verified")
    
    print("\n[2] Setting up default admin user...")
    
    # Check if any admin users exist
    all_users = db_manager.get_all_auth_users()
    admin_exists = any(user['role'] == 'admin' for user in all_users)
    
    if admin_exists:
        print("    ℹ Admin user already exists")
        response = input("    Create another admin user? (y/n): ").strip().lower()
        if response != 'y':
            print("\n✓ Initialization complete!")
            return True
    
    # Create admin user
    print("\n    Enter admin credentials:")
    admin_username = input("    Admin username: ").strip()
    
    if not admin_username:
        print("    [!] Username cannot be empty")
        return False
    
    # Check if username already exists
    existing_user = db_manager.get_auth_user(admin_username)
    if existing_user:
        print(f"    [!] User '{admin_username}' already exists")
        return False
    
    admin_password = getpass("    Admin password: ")
    admin_password_confirm = getpass("    Confirm password: ")
    
    if admin_password != admin_password_confirm:
        print("    [!] Passwords do not match")
        return False
    
    if len(admin_password) < 6:
        print("    [!] Password must be at least 6 characters")
        return False
    
    admin_display_name = input("    Display name (optional): ").strip() or admin_username
    
    # Hash password and create admin user
    password_hash = hash_password(admin_password)
    success = db_manager.create_auth_user(
        username=admin_username,
        password_hash=password_hash,
        role='admin',
        display_name=admin_display_name
    )
    
    if success:
        print(f"\n    ✓ Admin user '{admin_username}' created successfully")
    else:
        print(f"\n    [!] Failed to create admin user")
        return False
    
    print("\n" + "=" * 60)
    print("✓ Authentication database initialized successfully!")
    print("=" * 60)
    print(f"\nAdmin credentials:")
    print(f"  Username: {admin_username}")
    print(f"  Password: {'*' * len(admin_password)}")
    print(f"\nYou can now start the Flask server and log in as admin.")
    print("=" * 60)
    
    return True


def create_test_users():
    """Create test users for development"""
    print("\n[Optional] Create test users for development?")
    response = input("Create test users? (y/n): ").strip().lower()
    
    if response != 'y':
        return
    
    test_users = [
        {"username": "testuser1", "password": "password123", "display_name": "Test User 1"},
        {"username": "testuser2", "password": "password123", "display_name": "Test User 2"},
    ]
    
    print("\nCreating test users...")
    for user_data in test_users:
        password_hash = hash_password(user_data['password'])
        success = db_manager.create_auth_user(
            username=user_data['username'],
            password_hash=password_hash,
            role='user',
            display_name=user_data['display_name']
        )
        
        if success:
            print(f"  ✓ Created: {user_data['username']} (password: {user_data['password']})")
        else:
            print(f"  ℹ Skipped: {user_data['username']} (already exists)")


if __name__ == "__main__":
    print("\nStarting authentication database initialization...")
    print("Loading environment variables from .env file...\n")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("[!] Warning: .env file not found in current directory")
        print("    Make sure you're running this script from the facial_recong/ directory")
        print("    Or create a .env file with DATABASE_URL and other configuration\n")
    else:
        print("[+] .env file found\n")
    
    # Verify DATABASE_URL is set
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("[!] Error: DATABASE_URL not found in environment variables")
        print("    Please add DATABASE_URL to your .env file")
        print("    Example: DATABASE_URL=postgresql://user:pass@host/database\n")
        sys.exit(1)
    else:
        print(f"[+] Database URL configured: {db_url[:30]}...\n")
    
    try:
        success = init_auth_database()
        
        if success:
            create_test_users()
        else:
            print("\n[!] Initialization failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n[!] Initialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

