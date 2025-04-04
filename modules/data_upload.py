import streamlit as st
import pandas as pd

def apply_dark_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    apply_dark_theme()
    st.title("📤 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success("Data uploaded successfully!")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")