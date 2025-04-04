import streamlit as st
import os
import pathlib
import requests
import json
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests

# Read config from Streamlit secrets
GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = st.secrets["REDIRECT_URI"]

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]
AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USER_INFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"

# Enable insecure transport for local testing
if st.secrets.get("environment") == "local":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Path to temporary client secret file
CLIENT_SECRET_FILE = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

def create_client_secret_file():
    """Create Google client secret file dynamically"""
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "project_id": "business-analytics",
            "auth_uri": AUTH_URL,
            "token_uri": TOKEN_URL,
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uris": [REDIRECT_URI]
        }
    }
    try:
        with open(CLIENT_SECRET_FILE, 'w') as f:
            json.dump(client_config, f)
        return True
    except Exception as e:
        st.error(f"Error creating client_secret.json: {e}")
        return False

def setup_google_auth():
    """Initialize Google OAuth flow"""
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

def get_google_auth_url():
    """Generate login URL"""
    flow = setup_google_auth()
    if not flow:
        return None
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    return auth_url

# List of authorized users
ALLOWED_USERS = {
    "durai.varshith@gmail.com",
    "vishwajith@student.tce.edu",
    "duraisamy@student.tce.edu",
    "pnandhini@student.tce.edu",
    "sowmyashri@student.tce.edu",
    "krithikaa@student.tce.edu"
}

def process_callback(auth_code):
    """Handle the callback and fetch user info"""
    flow = setup_google_auth()
    if not flow:
        return None

    try:
        flow.fetch_token(code=auth_code)
        credentials = flow.credentials

        session = requests.session()
        cached_session = cachecontrol.CacheControl(session)
        token_request = google.auth.transport.requests.Request(session=cached_session)

        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            request=token_request,
            audience=GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )

        headers = {'Authorization': f'Bearer {credentials.token}'}
        response = requests.get(USER_INFO_URL, headers=headers)
        user_info = response.json()

        user_email = user_info.get("email", "").lower()
        if user_email not in ALLOWED_USERS:
            st.error("❌ Access denied. Unauthorized email.")
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
        st.error(f"⚠️ Authentication failed: {e}")
        return None

import streamlit as st

def show_login_page():
    """Render minimal login interface with Google Sign-in button"""
    
    # Custom CSS for styling
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
            
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #fff;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            
            .login-container {
                text-align: center;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                width: 100%;
            }
            
            /* Google Sign-in Button */
            .google-signin {
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #4285F4;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                font-weight: 500;
                cursor: pointer;
                width: 100%;
                text-decoration: none;
                transition: background-color 0.3s;
            }
            
            .google-signin img {
                width: 20px;
                height: 20px;
                margin-right: 12px;
                background-color: white;
                padding: 4px;
                border-radius: 2px;
            }
            
            .google-signin:hover {
                background-color: #357ae8;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Google Sign-in Button
    auth_url = get_google_auth_url()
    if auth_url:
        st.markdown(f"""
            <a href="{auth_url}" class="google-signin">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg">
                Sign in with Google
            </a>
        """, unsafe_allow_html=True)
    else:
        st.error("Failed to create Google login link.")
    
    # Closing tag
    st.markdown('</div>', unsafe_allow_html=True)
    

def logout():
    """Clear session state"""
    st.session_state.clear()

def get_user_info():
    """Get logged-in user's info"""
    return st.session_state.get("user_info", None)

# Ensure auth state is initialized
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
