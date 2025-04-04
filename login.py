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
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Only for local development

# ✅ Allowed emails for login
ALLOWED_USERS = {
    "durai.varshith@gmail.com",
    "vishwajith@student.tce.edu",
    "duraisamy@student.tce.edu"
}

# 🔧 Create client_secret.json dynamically
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

# 🚀 Setup OAuth flow
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

# 🔗 Get auth URL
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

# 🔁 Process callback after redirect from Google
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
            st.error("🚫 Access denied. Your email is not authorized.")
            return None

        # ✅ Store in session
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

# 🔐 Login UI & flow
def show_login_page():
    st.title("🔐 EDA Dashboard Login")

    # ✅ If already authenticated, show welcome
    if st.session_state.get("authenticated", False):
        user_info = get_user_info()
        st.success(f"Welcome back, {user_info.get('name', 'User')}!")
        return True

    # 🌀 Process Google callback
    if "code" in st.query_params:
        with st.spinner("Authenticating..."):
            user_info = process_callback(st.query_params["code"])
            if user_info:
                st.query_params.clear()  # remove ?code= after login
                return True

    # 👤 Show login UI
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.3); border-radius: 10px;'>
            <h3>Sign in to access the Dashboard</h3>
            <p>Use your authorized Google account to continue.</p>
    """, unsafe_allow_html=True)

    auth_url = get_google_auth_url()
    if auth_url:
        st.markdown(f"""
        <a href="{auth_url}" target="_self">
            <button style="background-color:#4285F4; color:white; border:none; border-radius:5px; padding:10px 20px; font-size:16px; cursor:pointer;">Sign in with Google</button>
        </a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Could not generate authentication URL.")

    return False

# 🔓 Logout
def logout():
    st.session_state.clear()

# 👤 Get user info
def get_user_info():
    return st.session_state.get("user_info")
