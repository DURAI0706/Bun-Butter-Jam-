import streamlit as st
import os
import pathlib
import requests
import json
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests

# Load configuration from Streamlit secrets
def get_config(key: str, default=None):
    return st.secrets.get(key, default)

GOOGLE_CLIENT_ID = get_config("google_client_id")
GOOGLE_CLIENT_SECRET = get_config("google_client_secret")
REDIRECT_URI = get_config("redirect_uri_local") if get_config("environment") == "local" else get_config("redirect_uri_prod")

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]
AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USER_INFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"

CLIENT_SECRET_FILE = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # For local development only

# Create client secret file dynamically
def create_client_secret_file():
    config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "project_id": "business-analytics",
            "auth_uri": AUTH_URL,
            "token_uri": TOKEN_URL,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uris": [REDIRECT_URI],
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
        }
    }
    try:
        with open(CLIENT_SECRET_FILE, 'w') as f:
            json.dump(config, f)
        return True
    except Exception as e:
        st.error(f"Failed to write client_secret.json: {e}")
        return False

# Setup OAuth flow
def setup_google_auth():
    if not os.path.exists(CLIENT_SECRET_FILE):
        if not create_client_secret_file():
            return None
    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRET_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        return flow
    except Exception as e:
        st.error(f"OAuth setup failed: {e}")
        return None

# Get authorization URL
def get_google_auth_url():
    flow = setup_google_auth()
    if not flow:
        return None
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    return auth_url

# Allowed user emails
ALLOWED_USERS = {
    "durai.varshith@gmail.com",
    "vishwajith@student.tce.edu",
    "duraisamy@student.tce.edu"
}

# Handle callback
def process_callback(auth_code):
    try:
        flow = setup_google_auth()
        if not flow:
            return None
        flow.fetch_token(code=auth_code)
        credentials = flow.credentials

        session = requests.session()
        cached_session = cachecontrol.CacheControl(session)
        request = google.auth.transport.requests.Request(session=cached_session)

        id_info = id_token.verify_oauth2_token(
            id_token=credentials.id_token,
            request=request,
            audience=GOOGLE_CLIENT_ID
        )

        headers = {'Authorization': f'Bearer {credentials.token}'}
        user_info = requests.get(USER_INFO_URL, headers=headers).json()

        email = user_info.get("email", "").lower()
        if email not in ALLOWED_USERS:
            st.error("Access denied. Email not authorized.")
            return None

        st.session_state.update({
            "authenticated": True,
            "user_info": user_info,
            "credentials": {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'id_token': credentials.id_token,
                'expiry': credentials.expiry.isoformat() if credentials.expiry else None
            }
        })
        return user_info

    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None

def show_login_page():
    st.markdown("""
        <style>
            body {
                background-color: #121212;
            }
            .login-container {
                max-width: 400px;
                margin: auto;
                padding: 40px;
                background: #1E1E1E;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
                text-align: center;
            }
            .login-container h2 {
                color: white;
            }
            .login-button {
                background-color: #FF5733;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                margin-top: 10px;
            }
            .google-button {
                background-color: #4285F4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='login-container'>
            <h2>Log In to Your Account</h2>
            <input type='text' placeholder='Email' style='width: 100%; padding: 10px; margin-top: 10px; border-radius: 5px; border: none;'>
            <input type='password' placeholder='Password' style='width: 100%; padding: 10px; margin-top: 10px; border-radius: 5px; border: none;'>
            <button class='login-button'>Log In</button>
            <p style='color: white; margin-top: 10px;'>OR</p>
            <a href='""" + get_google_auth_url() + """'><button class='google-button'>Log In with Google</button></a>
        </div>
    """, unsafe_allow_html=True)

def logout():
    st.session_state.clear()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

show_login_page()

# Get current user info
def get_user_info():
    return st.session_state.get("user_info")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
