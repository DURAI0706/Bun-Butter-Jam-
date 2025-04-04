import streamlit as st
import os
from login import show_login_page, logout
from modules.navigation import main as navigation_main  # Importing navigation module

st.set_page_config(page_title="🧁 Coronation Bakery Sales Analytics", layout="wide")

def create_config_file():
    """Creates a default config file if it doesn't exist."""
    if not os.path.exists('config.yaml'):
        with open('config.yaml', 'w') as f:
            f.write("default_config: true")

def main():
    """Main function to handle authentication and navigation redirection."""
    create_config_file()  # Ensure config file exists

    # Initialize session state variables
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Show login page if not authenticated
    if not st.session_state.authenticated:
        authenticated = show_login_page()
        if authenticated:
            st.session_state.authenticated = True
            st.rerun()  # Force rerun to load the navigation page
        return  # Stop execution to prevent running navigation before rerun

    # If authenticated, load navigation module
    st.sidebar.button("Logout", on_click=logout)  # Logout button
    navigation_main()  # Run navigation.py main function

if __name__ == "__main__":
    main()
