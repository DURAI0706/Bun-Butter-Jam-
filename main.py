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
        st.title("📊 Dashboard")

        # Call your navigation function
        navigation_main()

        # Spacer to push the logout button down
        st.markdown('<div style="height: 300px;"></div>', unsafe_allow_html=True)

        # Styled logout button
        st.markdown("""
            <style>
                .logout-button {
                    position: fixed;
                    bottom: 20px;
                    left: 1.5rem;
                    width: 85%;
                }
                .logout-button button {
                    background-color: #FF4B4B;
                    color: white;
                    border: none;
                    padding: 0.5rem;
                    border-radius: 8px;
                    font-weight: bold;
                    width: 100%;
                }
                .logout-button button:hover {
                    background-color: #FF1E1E;
                    cursor: pointer;
                }
            </style>
            <div class="logout-button">
        """, unsafe_allow_html=True)
        st.button("🚪 Logout", on_click=logout, key="logout-btn")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
