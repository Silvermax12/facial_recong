import os
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Tuple, Optional
import json

class DatabaseManager:
    def __init__(self):
        # Render PostgreSQL connection
        self.connection_string = os.getenv('DATABASE_URL')
        if not self.connection_string:
            print("[!] DATABASE_URL not found in environment variables")
            self.connection = None
        else:
            try:
                self.connection = psycopg2.connect(self.connection_string)
                print("[+] Connected to Render PostgreSQL database")
                self.create_tables()
            except Exception as e:
                print(f"[!] Database connection failed: {e}")
                self.connection = None

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.connection:
            return

        try:
            with self.connection.cursor() as cursor:
                # Auth users table (for authentication)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auth_users (
                        username VARCHAR(50) PRIMARY KEY,
                        password_hash VARCHAR(255) NOT NULL,
                        role VARCHAR(20) DEFAULT 'user' NOT NULL,
                        allowed BOOLEAN DEFAULT TRUE NOT NULL,
                        display_name VARCHAR(100),
                        skip_challenges BOOLEAN DEFAULT FALSE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                        last_login TIMESTAMP
                    );
                """)

                # Users table (for face enrollment tracking)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username VARCHAR(255) PRIMARY KEY,
                        enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        face_count INTEGER DEFAULT 0,
                        last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Face encodings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_encodings (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) REFERENCES users(username) ON DELETE CASCADE,
                        encoding_vector DOUBLE PRECISION[],  -- Array of 128 floats
                        cloudinary_url VARCHAR(500),
                        enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Create index for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_face_encodings_username
                    ON face_encodings(username);
                """)

                # Migration: Add face_count column if it doesn't exist
                cursor.execute("""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='users' AND column_name='face_count'
                        ) THEN
                            ALTER TABLE users ADD COLUMN face_count INTEGER DEFAULT 0;
                        END IF;
                    END $$;
                """)
                
                # Migration: Add last_verified column if it doesn't exist
                cursor.execute("""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='users' AND column_name='last_verified'
                        ) THEN
                            ALTER TABLE users ADD COLUMN last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                        END IF;
                    END $$;
                """)
                
                # Migration: Remove old auth columns from users table if they exist
                # (password_hash, role, allowed, display_name, created_at, last_login)
                cursor.execute("""
                    DO $$
                    BEGIN
                        -- Drop password_hash column if it exists
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='users' AND column_name='password_hash'
                        ) THEN
                            ALTER TABLE users DROP COLUMN password_hash;
                        END IF;

                        -- Drop role column if it exists
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='users' AND column_name='role'
                        ) THEN
                            ALTER TABLE users DROP COLUMN role;
                        END IF;

                        -- Drop allowed column if it exists
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='users' AND column_name='allowed'
                        ) THEN
                            ALTER TABLE users DROP COLUMN allowed;
                        END IF;

                        -- Drop display_name column if it exists
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='users' AND column_name='display_name'
                        ) THEN
                            ALTER TABLE users DROP COLUMN display_name;
                        END IF;

                        -- Drop created_at column if it exists
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='users' AND column_name='created_at'
                        ) THEN
                            ALTER TABLE users DROP COLUMN created_at;
                        END IF;

                        -- Drop last_login column if it exists
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='users' AND column_name='last_login'
                        ) THEN
                            ALTER TABLE users DROP COLUMN last_login;
                        END IF;
                    END $$;
                """)

                # Migration: Add skip_challenges column to auth_users if it doesn't exist
                cursor.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='auth_users' AND column_name='skip_challenges'
                        ) THEN
                            ALTER TABLE auth_users ADD COLUMN skip_challenges BOOLEAN DEFAULT FALSE NOT NULL;
                        END IF;
                    END $$;
                """)

                self.connection.commit()
                print("[+] Database tables created successfully")
                print("[+] Applied migrations: face_count, last_verified, skip_challenges, removed legacy auth columns")

        except Exception as e:
            print(f"[!] Failed to create tables: {e}")
            self.connection.rollback()

    def enroll_user(self, username: str, face_encodings: List[np.ndarray],
                   cloudinary_urls: List[str]) -> bool:
        """Enroll a user with their face encodings and photo URLs"""
        if not self.connection:
            print("[!] No database connection")
            return False

        try:
            with self.connection.cursor() as cursor:
                # Insert or update user
                cursor.execute("""
                    INSERT INTO users (username, face_count)
                    VALUES (%s, %s)
                    ON CONFLICT (username)
                    DO UPDATE SET face_count = EXCLUDED.face_count,
                                  last_verified = CURRENT_TIMESTAMP;
                """, (username, len(face_encodings)))

                # Insert face encodings
                for i, (encoding, url) in enumerate(zip(face_encodings, cloudinary_urls)):
                    # Convert numpy array to list for PostgreSQL
                    encoding_list = encoding.tolist() if isinstance(encoding, np.ndarray) else encoding
                    cursor.execute("""
                        INSERT INTO face_encodings (username, encoding_vector, cloudinary_url)
                        VALUES (%s, %s, %s);
                    """, (username, encoding_list, url))

                self.connection.commit()
                print(f"[+] Enrolled user '{username}' with {len(face_encodings)} faces")
                return True

        except Exception as e:
            print(f"[!] Enrollment failed: {e}")
            self.connection.rollback()
            return False

    def get_user_encodings(self, username: str) -> List[Tuple[np.ndarray, str]]:
        """Get all face encodings and URLs for a user"""
        if not self.connection:
            return []

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT encoding_vector, cloudinary_url
                    FROM face_encodings
                    WHERE username = %s
                    ORDER BY enrolled_at;
                """, (username,))

                results = cursor.fetchall()
                encodings_and_urls = []

                for row in results:
                    # Convert list back to numpy array
                    encoding = np.array(row['encoding_vector'], dtype=np.float64)
                    url = row['cloudinary_url']
                    encodings_and_urls.append((encoding, url))

                return encodings_and_urls

        except Exception as e:
            print(f"[!] Failed to get user encodings: {e}")
            return []

    def get_all_users(self) -> List[str]:
        """Get list of all enrolled users"""
        if not self.connection:
            return []

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT username FROM users ORDER BY enrolled_at;")
                results = cursor.fetchall()
                return [row[0] for row in results]

        except Exception as e:
            print(f"[!] Failed to get users: {e}")
            return []

    def update_last_verified(self, username: str):
        """Update the last verified timestamp for a user"""
        if not self.connection:
            return

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET last_verified = CURRENT_TIMESTAMP
                    WHERE username = %s;
                """, (username,))
                self.connection.commit()

        except Exception as e:
            print(f"[!] Failed to update last verified: {e}")
            self.connection.rollback()

    def get_user_stats(self, username: str) -> dict:
        """Get statistics for a user"""
        if not self.connection:
            return {}

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT u.username, u.enrolled_at, u.face_count, u.last_verified,
                           COUNT(fe.id) as current_faces
                    FROM users u
                    LEFT JOIN face_encodings fe ON u.username = fe.username
                    WHERE u.username = %s
                    GROUP BY u.username, u.enrolled_at, u.face_count, u.last_verified;
                """, (username,))

                result = cursor.fetchone()
                return dict(result) if result else {}

        except Exception as e:
            print(f"[!] Failed to get user stats: {e}")
            return {}

    # ========================================
    # Authentication Methods
    # ========================================

    def create_auth_user(self, username: str, password_hash: str, role: str = 'user', 
                        display_name: Optional[str] = None) -> bool:
        """Create a new auth user"""
        if not self.connection:
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO auth_users (username, password_hash, role, display_name, allowed)
                    VALUES (%s, %s, %s, %s, TRUE)
                """, (username, password_hash, role, display_name or username))
                self.connection.commit()
                print(f"[+] Created auth user '{username}' with role '{role}'")
                return True

        except psycopg2.IntegrityError:
            print(f"[!] User '{username}' already exists")
            self.connection.rollback()
            return False
        except Exception as e:
            print(f"[!] Failed to create auth user: {e}")
            self.connection.rollback()
            return False

    def get_auth_user(self, username: str) -> Optional[dict]:
        """Get auth user by username"""
        if not self.connection:
            return None

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT username, password_hash, role, allowed, display_name,
                           skip_challenges, created_at, last_login
                    FROM auth_users
                    WHERE username = %s;
                """, (username,))

                result = cursor.fetchone()
                return dict(result) if result else None

        except Exception as e:
            print(f"[!] Failed to get auth user: {e}")
            return None

    def update_last_login(self, username: str):
        """Update last login timestamp"""
        if not self.connection:
            return

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE auth_users
                    SET last_login = CURRENT_TIMESTAMP
                    WHERE username = %s;
                """, (username,))
                self.connection.commit()

        except Exception as e:
            print(f"[!] Failed to update last login: {e}")
            self.connection.rollback()

    def is_user_allowed(self, username: str) -> bool:
        """Check if user is allowed (not blocked)"""
        if not self.connection:
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT allowed FROM auth_users WHERE username = %s;
                """, (username,))
                result = cursor.fetchone()
                return result[0] if result else False

        except Exception as e:
            print(f"[!] Failed to check user status: {e}")
            return False

    def set_user_allowed(self, username: str, allowed: bool) -> bool:
        """Set user allowed status (block/unblock)"""
        if not self.connection:
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE auth_users
                    SET allowed = %s
                    WHERE username = %s;
                """, (allowed, username))
                self.connection.commit()
                status = "allowed" if allowed else "blocked"
                print(f"[+] User '{username}' {status}")
                return True

        except Exception as e:
            print(f"[!] Failed to update user status: {e}")
            self.connection.rollback()
            return False

    def get_all_auth_users(self) -> List[dict]:
        """Get all auth users (for admin dashboard)"""
        if not self.connection:
            return []

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT username, role, allowed, display_name, skip_challenges, created_at, last_login
                    FROM auth_users
                    ORDER BY created_at DESC;
                """)
                results = cursor.fetchall()
                return [dict(row) for row in results]

        except Exception as e:
            print(f"[!] Failed to get all auth users: {e}")
            return []

    def set_user_skip_challenges(self, username: str, skip_challenges: bool) -> bool:
        """Set user skip_challenges flag"""
        if not self.connection:
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE auth_users
                    SET skip_challenges = %s
                    WHERE username = %s;
                """, (skip_challenges, username))
                self.connection.commit()
                status = "enabled" if skip_challenges else "disabled"
                print(f"[+] Challenge skipping {status} for user '{username}'")
                return True

        except Exception as e:
            print(f"[!] Failed to update user challenge settings: {e}")
            self.connection.rollback()
            return False

    def get_user_skip_challenges(self, username: str) -> bool:
        """Check if user should skip challenges"""
        if not self.connection:
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT skip_challenges FROM auth_users WHERE username = %s;
                """, (username,))
                result = cursor.fetchone()
                return result[0] if result else False

        except Exception as e:
            print(f"[!] Failed to check user challenge settings: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("[+] Database connection closed")

# Global database manager instance
db_manager = DatabaseManager()
