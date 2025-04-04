import streamlit as st
import os
from login import show_login_page, logout
from modules.navigation import main as navigation_main  # Importing navigation module

st.set_page_config(page_title="🧁 Coronation Bakery Sales Analytics", layout="wide")

def create_config_file():
    if not os.path.exists('config.yaml'):
        with open('config.yaml', 'w') as f:
            f.write("default_config: true")

def main():
    create_config_file()

    # Initialize only once
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # ✅ Keep user logged in if already authenticated
    if not st.session_state.authenticated:
        if show_login_page():  # This handles login + sets session state
            st.session_state.authenticated = True
            st.rerun()
        return  # Prevent app from loading until logged in

    # ✅ Authenticated state: Show dashboard
    st.sidebar.button("Logout", on_click=logout)
    navigation_main()

if __name__ == "__main__":
    main()
