import streamlit as st
import os
import pathlib
import requests
import json
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests

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

ALLOWED_USERS = {
    "durai.varshith@gmail.com",
    "vishwajith@student.tce.edu",
    "duraisamy@student.tce.edu",
    "pnandhini@student.tce.edu",
    "sowmyashri@student.tce.edu",
    "krithikaa@student.tce.edu"
}

def get_google_auth_url():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    return auth_url

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


def show_login_page():
    """Render login interface and process login"""
    st.title("🔐 Login to Coronation Bakery Dashboard")

    # Handle redirect callback
    if "code" in st.query_params:
        with st.spinner("Authenticating..."):
            user_info = process_callback(st.query_params["code"])
            if user_info:
                st.success(f"🎉 Welcome, {user_info.get('name', 'User')}!")
                st.query_params.clear()
                return True

    # Show sign-in button
    if not st.session_state.get("authenticated", False):
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>Please sign in with your Google account to continue</h3>
        </div>
        """, unsafe_allow_html=True)

        auth_url = get_google_auth_url()
        if auth_url:
            st.markdown(f"""
                <a href="{auth_url}">
                    <button style="background-color:#4285F4;color:white;padding:10px 20px;border:none;border-radius:5px;font-size:16px;">
                        Sign in with Google
                    </button>
                </a>
            """, unsafe_allow_html=True)
        else:
            st.error("🚨 Failed to create Google login link.")

    return st.session_state.get("authenticated", False)

def logout():
    """Clear session state"""
    st.session_state.clear()

def get_user_info():
    """Get logged-in user's info"""
    return st.session_state.get("user_info", None)

# Ensure auth state is initialized
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
