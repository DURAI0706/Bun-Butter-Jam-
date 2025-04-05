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
    st.sidebar.button("Logout", on_click=logout)
    navigation_main()

if __name__ == "__main__":
    main()
