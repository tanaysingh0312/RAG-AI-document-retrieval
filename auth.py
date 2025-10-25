import bcrypt
import sqlite3
import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Import JWT utilities
from jose import jwt, JWTError

# ==============================================================================
# ====== CONFIGURATION SETTINGS (Shared/Adjusted) ======
# ==============================================================================
# Database paths relative to where the FastAPI app will be run (e.g., from pyrotech_fastapi_rag directory)
DB_DIR = "db" # This will be C:\Users\PTPLAI\chatbot-env\db
USERS_DB = os.path.join(DB_DIR, "users.db")
CHAT_LOGS_DB = os.path.join(DB_DIR, "chat_logs.db")

# JWT Configuration
# THIS IS CRITICAL FOR SECURITY - DO NOT HARDCODE IN PRODUCTION
# It's better to read this from an environment variable in main.py and pass it,
# but for auth.py to be self-contained for JWT functions, we'll set a default.
# In main.py, you will still define and use SECRET_KEY from env.
_AUTH_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-please-change-this")
_AUTH_ALGORITHM = "HS256"
_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES = 30 # For user login sessions


# ==============================================================================
# ====== Pydantic Models for Authentication & Chat History (DEFINITIVE) ======
# ==============================================================================
class User(BaseModel):
    """Pydantic model for a user, used in authentication dependencies."""
    username: str
    role: str = "user" # Default role, can be 'admin'

class SessionSummary(BaseModel):
    """Pydantic model for a chat session summary (for listing user sessions)."""
    session_id: str
    session_title: Optional[str] = None
    last_message_timestamp: Optional[str] = None # Added for session listing

class SessionMessages(BaseModel):
    """Pydantic model for retrieving full messages of a specific session."""
    session_id: str
    session_title: Optional[str] = None
    messages: List[Dict] # List of dicts, as returned by auth.get_chat_messages_for_session

class AdminChatLogEntry(BaseModel):
    """Pydantic model for a single chat log entry, for admin viewing."""
    id: int
    username: str
    session_id: str
    session_title: Optional[str] = None
    role: str
    message: str
    timestamp: str


# ==============================================================================
# ====== PASSWORD HASHING (Using bcrypt for security) ======
# ==============================================================================

def hash_password(password_str: str) -> str:
    """Hashes a password using bcrypt for secure storage."""
    return bcrypt.hashpw(password_str.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password_str: str, hashed_str: str) -> bool:
    """Verifies a plaintext password against a hashed password using bcrypt."""
    try:
        if hashed_str and (hashed_str.startswith('$2b$') or hashed_str.startswith('$2a$') or hashed_str.startswith('$2y$')):
            return bcrypt.checkpw(password_str.encode('utf-8'), hashed_str.encode('utf-8'))
        else:
            print("WARNING: Provided hash does not appear to be a bcrypt hash. Assuming mismatch.")
            return False
    except ValueError as e:
        print(f"ERROR: Invalid hash format during password check: {e}")
        return False
    except Exception as e:
        print(f"ERROR during password check: {e}")
        traceback.print_exc()
        return False

# ==============================================================================
# ====== DATABASE (SQLite for Users & Chat Logs) FUNCTIONS ======
# ==============================================================================

def init_sqlite_db(db_filename: str, table_schema: str):
    """Initializes or verifies a SQLite database and table, with schema migration for users."""
    print(f"INFO: Initializing SQLite database: '{db_filename}'...")
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        
        with sqlite3.connect(db_filename) as conn:
            cursor = conn.cursor()

            if "users.db" in db_filename:
                # Check if 'role' column exists in users table
                cursor.execute("PRAGMA table_info(users)")
                columns = [col[1] for col in cursor.fetchall()]
                if "role" not in columns:
                    print(f"INFO: Adding 'role' column to 'users' table in {db_filename}...")
                    try:
                        conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
                        conn.commit()
                        print("✅ 'role' column added to 'users' table.")
                    except sqlite3.OperationalError as e:
                        print(f"WARNING: Could not add 'role' column to 'users' table (might already exist or other issue): {e}")
                
                # Ensure the users table itself is created if it doesn't exist already
                conn.execute(table_schema)
                
                # Ensure an 'admin' user exists for initial setup
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
                if cursor.fetchone()[0] == 0:
                    admin_password_hash = hash_password("jahnavi@123") # Default password for admin
                    conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", ('admin', admin_password_hash, 'admin'))
                    conn.commit()
                    print("✅ Default 'admin' user created with password 'jahnavi@123'. IMPORTANT: Change this in production!")
                    print(f"DEBUG: Admin user 'admin' inserted into users.db with hash: {admin_password_hash[:10]}...")

            elif "chat_logs.db" in db_filename:
                # Ensure chat_message_logs table schema and index are correct
                conn.execute(table_schema)
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_message_logs (username, session_id)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_username ON chat_message_logs (username)")
                except sqlite3.OperationalError as e:
                    print(f"WARNING: Could not create index on chat_message_logs: {e}")
            
        print(f"✅ SQLite Database '{db_filename}' initialized/verified.")

    except Exception as e:
        print(f"❌ ERROR initializing SQLite database '{db_filename}': {e}")
        traceback.print_exc()
        raise # Re-raise to indicate critical startup failure

# ==============================================================================
# ====== USER MANAGEMENT FUNCTIONS ======
# ==============================================================================

def register_user(username_str: str, password_str: str, role: str = "user") -> tuple[bool, str]:
    """
    Registers a new user in the database.
    Returns (True, message) on success, (False, message) on failure.
    """
    if not username_str or not password_str:
        return False, "Username and password cannot be empty."
    
    target_role = role if role in ["user", "admin"] else "user"

    try:
        with sqlite3.connect(USERS_DB) as conn:
            conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                         (username_str, hash_password(password_str), target_role))
            conn.commit()
        print(f"INFO: User '{username_str}' registered successfully with role '{target_role}'.")
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        print(f"WARNING: Registration failed for user '{username_str}'. Username already exists.")
        return False, "Username already exists."
    except Exception as e:
        print(f"ERROR during user registration for '{username_str}': {e}")
        traceback.print_exc()
        return False, f"Registration error: {e}"

def get_user(username: str) -> Optional[User]:
    """Retrieves a user record from the database as a User Pydantic model."""
    try:
        with sqlite3.connect(USERS_DB) as conn:
            conn.row_factory = sqlite3.Row # Allows accessing columns by name
            result = conn.execute("SELECT username, role FROM users WHERE username = ?", (username,)).fetchone()
        if result:
            return User(username=result["username"], role=result["role"])
        return None # User not found
    except Exception as e:
        print(f"ERROR retrieving user '{username}': {e}")
        traceback.print_exc()
        return None

def authenticate_user(username_str: str, password_str: str) -> Optional[User]:
    """Authenticates a user against the database and returns a User object if successful."""
    try:
        with sqlite3.connect(USERS_DB) as conn:
            result = conn.execute("SELECT password_hash, role FROM users WHERE username = ?", (username_str,)).fetchone()
        if result and check_password(password_str, result[0]):
            print(f"INFO: User '{username_str}' authenticated successfully.")
            return User(username=username_str, role=result[1]) # Return User object
        print(f"WARNING: Authentication failed for user '{username_str}'. Invalid credentials.")
        return None
    except Exception as e:
        print(f"ERROR during user authentication for '{username_str}': {e}")
        traceback.print_exc()
        return None

def is_user_active(username: str) -> bool:
    """Placeholder for checking if a user is active."""
    return True # All users are active by default for this example

def get_user_role(username: str) -> Optional[str]:
    """Queries the users table to return the user's role."""
    try:
        with sqlite3.connect(USERS_DB) as conn:
            result = conn.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
        if result:
            return result[0]
        return None # User not found
    except Exception as e:
        print(f"ERROR retrieving role for user '{username}': {e}")
        traceback.print_exc()
        return None

def get_all_users() -> List[Dict]:
    """Returns a list of all registered users and their roles."""
    users = []
    try:
        with sqlite3.connect(USERS_DB) as conn:
            conn.row_factory = sqlite3.Row # Allows accessing columns by name
            cursor = conn.execute("SELECT username, role FROM users ORDER BY username ASC")
            for row in cursor.fetchall():
                users.append({"username": row["username"], "role": row["role"]})
    except Exception as e:
        print(f"ERROR retrieving all users: {e}")
        traceback.print_exc()
    return users

# ==============================================================================
# ====== JWT Token Creation & Verification FUNCTIONS (Moved to auth.py) ======
# ==============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, _AUTH_SECRET_KEY, _AUTH_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[User]:
    """Verifies a JWT token and returns the User object if valid."""
    try:
        payload = jwt.decode(token, _AUTH_SECRET_KEY, algorithms=[_AUTH_ALGORITHM])
        username: str = payload.get("sub")
        user_role: str = payload.get("role")
        if username is None or user_role is None:
            return None # Token does not contain expected user data
        
        user = get_user(username) # Get user from DB to ensure it's still valid/active
        if user is None:
            return None # User not found in DB

        return User(username=username, role=user_role)
    except JWTError as e:
        print(f"WARNING: JWT verification failed: {e}")
        return None # Invalid token


# ==============================================================================
# ====== CHAT SESSION MANAGEMENT FUNCTIONS ======
# ==============================================================================

def sanitize_text(text: Any) -> str:
    """Sanitizes text by replacing HTML special characters and stripping whitespace."""
    if text is None:
        return ""
    # Ensure text is treated as string for encoding
    s = str(text)
    # Replace invalid UTF-8 characters
    s = s.encode('utf-8', errors='replace').decode('utf-8')
    # Replace common HTML special chars. For full HTML sanitization, use a dedicated library.
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").strip()
    return s

def log_chat_message_to_db(username_str: str, session_id_str: str, role_str: str, message_content: str, session_title: Optional[str] = None):
    """
    Logs a single chat message (user or assistant) to the SQLite chat log database.
    Assumes `CHAT_LOGS_DB` is correctly imported or set as a fallback.
    """
    try:
        safe_message_content = sanitize_text(message_content)
        with sqlite3.connect(CHAT_LOGS_DB) as conn:
            cursor = conn.cursor()
            if session_title:
                sanitized_session_title = sanitize_text(session_title)
                # Only update session_title if it's the first message or a new title is provided for the session_id
                cursor.execute("""
                    UPDATE chat_message_logs
                    SET session_title = ?
                    WHERE username = ? AND session_id = ? AND session_title IS NULL
                """, (sanitized_session_title, username_str, session_id_str))
                conn.commit() # Commit the update before insert
            
            cursor.execute("""
                INSERT INTO chat_message_logs (username, session_id, session_title, role, message, timestamp)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (username_str, session_id_str, sanitize_text(session_title) if session_title else None, role_str, safe_message_content))
            conn.commit()
    except Exception as e:
        print(f"ERROR logging chat message to DB for user '{username_str}' session '{session_id_str}': {e}")
        traceback.print_exc()


def get_user_chat_sessions(username: str) -> List[SessionSummary]:
    """
    Retrieves distinct chat session summaries (session_id, title, last message timestamp) for a user.
    """
    try:
        with sqlite3.connect(CHAT_LOGS_DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT session_id, session_title, MAX(timestamp) as last_message_timestamp
                FROM chat_message_logs
                WHERE username = ?
                GROUP BY session_id, session_title
                ORDER BY last_message_timestamp DESC
            """, (username,))
            
            sessions_summary = []
            for row in cursor.fetchall():
                sessions_summary.append(SessionSummary(
                    session_id=row['session_id'],
                    session_title=row['session_title'] if row['session_title'] else f"Session {row['session_id'][:8]}",
                    last_message_timestamp=row['last_message_timestamp']
                ))
            return sessions_summary
    except Exception as e:
        print(f"ERROR listing sessions for user '{username}': {e}")
        traceback.print_exc()
        return []

def get_chat_messages_for_session(username: str, session_id: str) -> List[Dict]:
    """
    Retrieves all messages for a specific chat session of a user.
    """
    try:
        with sqlite3.connect(CHAT_LOGS_DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, username, session_id, session_title, role, message, timestamp
                FROM chat_message_logs
                WHERE username = ? AND session_id = ?
                ORDER BY timestamp ASC
            """, (username, session_id))
            messages = [{"id": row['id'], "username": row['username'], "session_id": row['session_id'],
                         "session_title": row['session_title'], "role": row['role'], "message": row['message'],
                         "timestamp": row['timestamp']} for row in cursor.fetchall()]
            return messages
    except Exception as e:
        print(f"ERROR retrieving messages for session '{session_id}' for user '{username}': {e}")
        traceback.print_exc()
        return []

def delete_user_session(session_id: str, username: Optional[str] = None, is_admin: bool = False) -> bool:
    """
    Deletes a specific chat session from the database.
    Admins (is_admin=True) can delete any session (username can be None).
    Regular users (is_admin=False) must provide their username and can only delete their own sessions.
    """
    try:
        with sqlite3.connect(CHAT_LOGS_DB) as conn:
            cursor = conn.cursor()
            if is_admin and username: # Admin deleting a specific user's session
                cursor.execute("DELETE FROM chat_message_logs WHERE username = ? AND session_id = ?", (username, session_id))
                print(f"INFO: Admin deleted session '{session_id}' for user '{username}'. Rows affected: {cursor.rowcount}")
            elif is_admin: # Admin deleting any session by session_id (e.g., if username is not known easily)
                cursor.execute("DELETE FROM chat_message_logs WHERE session_id = ?", (session_id,))
                print(f"INFO: Admin deleted session '{session_id}' (any user). Rows affected: {cursor.rowcount}")
            elif username: # Regular user deleting their own session
                cursor.execute("DELETE FROM chat_message_logs WHERE username = ? AND session_id = ?", (username, session_id))
                print(f"INFO: User '{username}' deleted session '{session_id}'. Rows affected: {cursor.rowcount}")
            else:
                print("ERROR: delete_user_session called without sufficient context (username for non-admin, or session_id).")
                return False
            
            conn.commit()
            return cursor.rowcount > 0 # Return True if any rows were deleted
    except Exception as e:
        print(f"ERROR deleting session '{session_id}' for user '{username}' (admin: {is_admin}): {e}")
        traceback.print_exc()
        return False

def get_all_chat_message_logs() -> List[Dict]:
    """
    Retrieves all chat message logs from the database for admin viewing.
    """
    logs = []
    try:
        with sqlite3.connect(CHAT_LOGS_DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT id, username, session_id, session_title, role, message, timestamp FROM chat_message_logs ORDER BY timestamp ASC")
            for row in cursor.fetchall():
                logs.append({k: row[k] for k in row.keys()}) # Convert Row object to dict
    except Exception as e:
        print(f"ERROR retrieving all chat logs: {e}")
        traceback.print_exc()
    return logs

def delete_old_chat_logs(days_old: int) -> bool:
    """
    Deletes chat messages older than a specified number of days.
    """
    try:
        threshold_date = datetime.now() - timedelta(days=days_old)
        threshold_timestamp = threshold_date.strftime('%Y-%m-%d %H:%M:%S')
        
        with sqlite3.connect(CHAT_LOGS_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_message_logs WHERE timestamp < ?", (threshold_timestamp,))
            conn.commit()
            print(f"INFO: Deleted {cursor.rowcount} chat logs older than {days_old} days.")
            return True
    except Exception as e:
        print(f"ERROR deleting old chat logs: {e}")
        traceback.print_exc()
        return False

# Initialize user and chat log databases when auth.py is imported
try:
    # Schema for users table (with role column)
    users_table_schema = """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """
    init_sqlite_db(USERS_DB, users_table_schema)

    # Schema for chat logs table
    chat_logs_table_schema = """
        CREATE TABLE IF NOT EXISTS chat_message_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            session_title TEXT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    init_sqlite_db(CHAT_LOGS_DB, chat_logs_table_schema)

except Exception as e:
    print(f"FATAL: One or more database initializations failed: {e}")
    traceback.print_exc()

