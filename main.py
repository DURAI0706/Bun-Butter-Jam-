
import streamlit as st
from login import show_login_page, logout
from modules.navigation import main as navigation_main

st.set_page_config(page_title="🧁 Coronation Bakery Sales Analytics", layout="wide")

def main():
    # Ensure session state key exists
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # If not logged in, show login
    if not st.session_state["authenticated"]:
        if show_login_page():
            st.session_state["authenticated"] = True
            st.rerun()
        return

    # Logged in, show dashboard
    with st.sidebar:
        navigation_main()  # Main navigation buttons or options
    
        # Add space to push logout button to bottom
        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
    
        # Logout button with icon at bottom
        st.markdown(
            """
            <style>
                .logout-button {
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    width: calc(100% - 40px);
                }
            </style>
            <div class="logout-button">
            """,
            unsafe_allow_html=True,
        )
        st.button("🚪 Logout", on_click=logout)

if __name__ == "__main__":
    main()
