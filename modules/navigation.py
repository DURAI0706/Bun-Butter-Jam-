import streamlit as st
from . import detail_eda  # Changed to relative import
from . import ml  # Changed to relative import


def main():
    st.sidebar.title("📊 Dashboard Modules")
    
    # Hamburger menu style
    st.markdown("""
    <style>
        div[data-testid="stSidebarNav"] {
            background-color: #1E2130;
        }
        div[data-testid="stSidebarNav"] ul {
            padding-left: 0;
        }
        div[data-testid="stSidebarNav"] li {
            list-style-type: none;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation options
    menu = st.sidebar.radio(
        "Select Module",
        ["🏠 Home", "📊 Detailed EDA", "🤖 ML Algorithms"],
        key="nav_menu"
    )
    
    # Route to selected module
    if menu == "🏠 Home":
        st.title("Dashboard Home")
        st.write("Welcome to the Analytics Dashboard. Select a module from the sidebar.")
    elif menu == "📊 Detailed EDA":
        detail_eda.main()
    elif menu == "🤖 ML Algorithms":
        ml.main()