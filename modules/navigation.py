import streamlit as st
from . import detail_eda  # Changed to relative import
from . import ml  # Changed to relative import
from . import home  # Changed to relative import
from streamlit_option_menu import option_menu

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
    selected = st.sidebar.option_menu(
        "Main Menu", 
        ["🏠 Home", "📊 Detailed EDA", "🤖 ML Algorithms"],
        menu_icon="cast", 
        default_index=0
    )
    
    # Route to selected module
    if menu == "🏠 Home":
        home.load_module()
    elif menu == "📊 Detailed EDA":
        detail_eda.main()
    elif menu == "🤖 ML Algorithms":
        ml.main()
