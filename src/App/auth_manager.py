import pymysql
import hashlib

# --- CONFIGURATION DIRECTE (Pour éviter les bugs d'import) ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "MAMAmavie21." 
DB_NAME = "PetroScope"

def get_connection():
    """Crée une connexion propre à la base"""
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME
    )

def hash_password(password):
    """Transforme le mot de passe en code secret illisible (SHA256)"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user(username, password):
    """Crée un nouvel utilisateur dans la BDD"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # On vérifie d'abord si l'utilisateur existe déjà
        cursor.execute("SELECT * FROM USERS WHERE username=%s", (username,))
        if cursor.fetchone():
            conn.close()
            return False # Déjà pris
        
        # Sinon on crée
        pwd_hash = hash_password(password)
        cursor.execute("INSERT INTO USERS (username, password_hash) VALUES (%s, %s)", (username, pwd_hash))
        conn.commit()
        conn.close()
        return True # Succès
    except Exception as e:
        print(f"Erreur création user : {e}")
        return False

def check_login(username, password):
    """Vérifie le login et le mot de passe"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        pwd_hash = hash_password(password)
        
        # On cherche l'utilisateur avec ce nom ET ce mot de passe haché
        cursor.execute("SELECT * FROM USERS WHERE username=%s AND password_hash=%s", (username, pwd_hash))
        result = cursor.fetchone()
        conn.close()
        
        return result is not None # Renvoie Vrai si trouvé, Faux sinon
    except Exception as e:
        print(f"Erreur login : {e}")
        return False

