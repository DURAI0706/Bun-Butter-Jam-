import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def load_module():
    st.markdown("<h1 style='text-align: center;'>Coronation Bakery Sales Dashboard</h1>", unsafe_allow_html=True)
    data = load_data()
    if data is not None:
        col_types = get_column_types(data)
        filtered_data = create_filters(data, col_types)
        display_metrics(filtered_data)
        display_charts(filtered_data)

def load_data():
    try:
        file_path = os.path.join('data', 'Coronation_Bakery_version_3.csv')
        if not os.path.exists(file_path):
            st.error(f"Data file not found at: {file_path}")
            st.info("Please ensure your CSV file is in the 'data' directory and named 'Coronation_Bakery_version_3.csv'")
            return None
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_column_types(df):
    col_types = {
        'numeric': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return col_types

def create_filters(df, col_types):
    """Create dynamic filters in sidebar"""
    st.sidebar.header("Filters")

    # Date filter
    date_col = df.select_dtypes(include=['datetime']).columns.tolist()
    if date_col:
        selected_date_col = st.sidebar.selectbox("Select date column for filtering", date_col)
        min_date = df[selected_date_col].min()
        max_date = df[selected_date_col].max()

        date_range = st.sidebar.date_input(
            f"Select date range for {selected_date_col}",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            df = df[(df[selected_date_col] >= start_date) & (df[selected_date_col] <= end_date)]

    # Numeric filters (first 3 only)
    for col in col_types['numeric'][:3]:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        values = st.sidebar.slider(
            f"Filter {col}",
            min_val, max_val, (min_val, max_val)
        )
        df = df[(df[col] >= values[0]) & (df[col] <= values[1])]

    # Categorical filters (first 3 only)
    for col in col_types['categorical'][:3]:
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(
            f"Filter {col}",
            options,
            default=options
        )
        if selected:
            df = df[df[col].isin(selected)]

    return df

def display_metrics(df):
    total_revenue = df['Total_Amount'].sum()
    total_sales = df['Quantity'].sum()
    days_count = len(df['Date'].dt.date.unique())
    avg_sales = total_sales / days_count if days_count > 0 else 0

    st.markdown("""
    <style>
    .metric-title { font-size: 17px !important; font-weight: 700 !important; margin-bottom: 5px !important; }
    .metric-value { font-size: 22px !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

    row = st.columns(3)
    metrics = [
        {"title": "Total Revenue", "value": f"₹{total_revenue:,.2f}"},
        {"title": "Total Sales Count", "value": f"{total_sales:,}"},
        {"title": "Average Sales/Day", "value": f"{avg_sales:,.1f}"}
    ]

    for i, col in enumerate(row):
        col.markdown(f"<p class='metric-title'>{metrics[i]['title']}</p>", unsafe_allow_html=True)
        col.markdown(f"<p class='metric-value'>{metrics[i]['value']}</p>", unsafe_allow_html=True)

def display_charts(df):
    # Row 1: Top Products & Seller Contribution
    row1 = st.columns(2)

    with row1[0]:
        st.subheader("Top 5 Best-Selling Products")
        top_products = df.groupby('Product_Type')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False).head(5)
        fig = go.Figure(go.Bar(
            y=top_products['Product_Type'],
            x=top_products['Quantity'],
            orientation='h',
            marker=dict(color='royalblue'),
            text=top_products['Quantity'],
            textposition='auto'
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with row1[1]:
        st.subheader("Sales Contribution by Seller")
        seller_sales = df.groupby(['Seller_Name', 'Product_Type'])['Quantity'].sum().reset_index()
        fig = px.sunburst(
            seller_sales,
            path=['Seller_Name', 'Product_Type'],
            values='Quantity',
            color='Seller_Name',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Daily Sales Trend
    st.subheader("Daily Sales Trend")
    daily_sales = df.groupby(df['Date'].dt.date)['Quantity'].sum().reset_index()
    fig = px.line(daily_sales, x='Date', y='Quantity', markers=True)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Row 3: Product Count by Seller
    st.subheader("Product Count by Seller")
    seller_product_counts = df.groupby(['Seller_Name', 'Product_Type'])['Quantity'].sum().reset_index()
    fig = px.bar(
        seller_product_counts,
        x='Seller_Name',
        y='Quantity',
        color='Product_Type',
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
