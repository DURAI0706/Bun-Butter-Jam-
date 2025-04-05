import streamlit as st
from login import show_login_page, logout
from modules.navigation import main as navigation_main  # Replace with your actual main function

st.set_page_config(page_title="🧁 Coronation Bakery Sales Analytics", layout="wide")

def main():
    # Initialize auth state once
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # If not logged in, show login page
    if not st.session_state.authenticated:
        authenticated = show_login_page()
        if authenticated:
            st.session_state.authenticated = True
            st.rerun()  # This ensures after login, we reload properly
        return

    # If already authenticated, stay on main page
    st.sidebar.button("Logout", on_click=logout)
    navigation_main()

if __name__ == "__main__":
    main()
