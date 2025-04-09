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
        create_filters(data)
        display_metrics(data)
        display_charts(data)

def load_data():
    try:
        file_path = os.path.join('data', 'Coronation_Bakery_version_3.csv')
        if not os.path.exists(file_path):
            st.error(f"Data file not found at: {file_path}")
            st.info("Please ensure your CSV file is in the 'data' directory and named 'Coronation_Bakery_version_3.csv'")
            return None
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        st.session_state['sales_data'] = df
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_filters(df):
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = df['Date'].min()
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = df['Date'].max()
    if 'selected_seller' not in st.session_state:
        st.session_state['selected_seller'] = "All Sellers"
    if 'selected_product' not in st.session_state:
        st.session_state['selected_product'] = "All Products"
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("Start Date", value=st.session_state['start_date'], min_value=df['Date'].min().to_pydatetime(), max_value=df['Date'].max().to_pydatetime(), format="DD-MM-YYYY")
        st.session_state['start_date'] = start_date
    with col2:
        end_date = st.date_input("End Date", value=st.session_state['end_date'], min_value=df['Date'].min().to_pydatetime(), max_value=df['Date'].max().to_pydatetime(), format="DD-MM-YYYY")
        st.session_state['end_date'] = end_date
        if end_date < start_date:
            st.error("End date must be after start date. Resetting to default range.")
            st.session_state['start_date'] = df['Date'].min()
            st.session_state['end_date'] = df['Date'].max()
    with col3:
        sellers = sorted(df['Seller_Name'].unique().tolist())
        seller_options = ["All Sellers"] + sellers
        selected_seller = st.selectbox("Select Seller", options=seller_options, index=seller_options.index(st.session_state['selected_seller']))
        st.session_state['selected_seller'] = selected_seller
    with col4:
        if selected_seller == "All Sellers":
            product_df = df
        else:
            product_df = df[df['Seller_Name'] == selected_seller]
        product_options = ["All Products"] + sorted(product_df['Product_Type'].unique().tolist())
        if st.session_state['selected_product'] not in product_options:
            st.session_state['selected_product'] = "All Products"
        selected_product = st.selectbox("Select Product Type", options=product_options, index=product_options.index(st.session_state['selected_product']))
        st.session_state['selected_product'] = selected_product

def filter_data(df):
    start_date = st.session_state['start_date']
    end_date = st.session_state['end_date']
    selected_seller = st.session_state['selected_seller']
    selected_product = st.session_state['selected_product']
    filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    if filtered_df.empty:
        st.warning("No data available for the selected date range. Please adjust your filter.")
        return df
    if selected_seller != "All Sellers":
        seller_filtered = filtered_df[filtered_df['Seller_Name'] == selected_seller]
        if seller_filtered.empty:
            st.warning(f"No data available for seller '{selected_seller}' in the selected date range.")
            return filtered_df
        filtered_df = seller_filtered
    if selected_product != "All Products":
        product_filtered = filtered_df[filtered_df['Product_Type'] == selected_product]
        if product_filtered.empty:
            st.warning(f"No data available for product '{selected_product}' with the current filters.")
            return filtered_df
        filtered_df = product_filtered
    return filtered_df

def display_metrics(df):
    filtered_df = filter_data(df)
    total_revenue = filtered_df['Total_Amount'].sum()
    total_sales = filtered_df['Quantity'].sum()
    days_count = len(filtered_df['Date'].dt.date.unique())
    avg_sales = total_sales / days_count if days_count > 0 else 0
    selected_seller = st.session_state['selected_seller']
    selected_product = st.session_state['selected_product']
    st.markdown("""
    <style>
    div.stContainer {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 8px;
        box-shadow: 0 0 8px rgba(0, 123, 255, 0.3);
        background-color: #F0F2F6;
        transition: box-shadow 0.3s ease;
        padding: 10px;
        text-align: center;
    }
    div.stContainer:hover {
        box-shadow: 0 0 12px rgba(0, 123, 255, 0.5);
    }
    .metric-title {
        font-size: 17px !important;
        font-weight: 700 !important;
        margin-bottom: 5px !important;
    }
    .metric-value {
        font-size: 22px !important;
        font-weight: 800 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    row = st.columns(5)
    metrics = [
        {"title": "Total Revenue", "value": f"₹{total_revenue:,.2f}"},
        {"title": "Total Sales Count", "value": f"{total_sales:,}"},
        {"title": "Average Sales/Day", "value": f"{avg_sales:,.1f}"},
        {"title": "Selected Seller", "value": selected_seller},
        {"title": "Selected Product", "value": selected_product}
    ]
    for i, col in enumerate(row):
        tile = col.container(height=120)
        tile.markdown(f"<p class='metric-title'>{metrics[i]['title']}</p>", unsafe_allow_html=True)
        tile.markdown(f"<p class='metric-value'>{metrics[i]['value']}</p>", unsafe_allow_html=True)

def display_charts(df):
    filtered_df = filter_data(df)
    
    # First row of charts
    row1 = st.columns(2)
    
    # Container 1: Top 5 Best-Selling Products Bar Chart
    with row1[0]:
        st.markdown("<h3 style='text-align: center;'>Top 5 Best-Selling Products</h3>", unsafe_allow_html=True)
        chart_container1 = st.container(height=300)
        with chart_container1:
            # Group by product and sum quantities
            product_sales = filtered_df.groupby('Product_Type')['Quantity'].sum().reset_index()
            # Sort by quantity in descending order and take top 5
            top_products = product_sales.sort_values('Quantity', ascending=False).head(5)
            
            if not top_products.empty:
                # Explicitly sort in descending order to ensure highest count appears first
                top_products = top_products.sort_values('Quantity', ascending=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=top_products['Product_Type'],
                    x=top_products['Quantity'],
                    orientation='h',
                    marker=dict(color='royalblue'),
                    text=top_products['Quantity'],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Quantity Sold",
                    font=dict(size=16),
                    xaxis=dict(showgrid=True),
                    yaxis=dict(
                        showgrid=False,
                        automargin=True,
                        tickfont=dict(size=14)
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No product data available for the selected filters.")
    
    
    # Container 2: Sales by Seller Pie Chart
    with row1[1]:
        st.markdown("<h3 style='text-align: center;'>Sales Contribution by Seller</h3>", unsafe_allow_html=True)
        chart_container2 = st.container(height=300)
        
        with chart_container2:
            # Filter based on date range
            temp_df = df[(df['Date'].dt.date >= st.session_state['start_date']) & 
                         (df['Date'].dt.date <= st.session_state['end_date'])]
        
            if not temp_df.empty:
                if 'Product_Type' in temp_df.columns:
                    # Group by Seller and Product_Type
                    seller_sales = temp_df.groupby(['Seller_Name', 'Product_Type'])['Quantity'].sum().reset_index()
                    
                    # Create path for hierarchical sunburst
                    seller_sales['path'] = seller_sales.apply(lambda row: f"{row['Seller_Name']}/{row['Product_Type']}", axis=1)
        
                    fig = px.sunburst(
                        seller_sales,
                        path=['Seller_Name', 'Product_Type'],
                        values='Quantity',
                        color='Seller_Name',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                else:
                    # If Product_Type not present, just show Seller breakdown
                    seller_sales = temp_df.groupby('Seller_Name')['Quantity'].sum().reset_index()
        
                    fig = px.sunburst(
                        seller_sales,
                        path=['Seller_Name'],
                        values='Quantity',
                        color='Seller_Name',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
        
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
        
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No seller data available for the selected date range.")

    
    # Second row of charts
    row2 = st.columns(1)
    
    # Container 3: Daily Sales Trend
    with row2[0]:
        st.markdown("<h3 style='text-align: center;'>Daily Sales Trend</h3>", unsafe_allow_html=True)
        chart_container3 = st.container(height=300)
        with chart_container3:
            daily_sales = filtered_df.groupby(filtered_df['Date'].dt.date)['Quantity'].sum().reset_index()
    
            if not daily_sales.empty and len(daily_sales) > 1:
                fig = px.line(
                    daily_sales, 
                    x='Date', 
                    y='Quantity',
                    markers=True
                )
    
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Date",
                    yaxis_title="Sales Quantity",
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough daily data available for trend chart.")
    
    row3 = st.columns(1)
    
    # Container 3: Daily Sales Trend
    with row3[0]:
        st.markdown("<h3 style='text-align: center;'>Product Count by Seller</h3>", unsafe_allow_html=True)
        chart_container4 = st.container(height=500)  # Increased height here
        
        with chart_container4:
            # Group by seller and product, then count
            seller_product_counts = filtered_df.groupby(['Seller_Name', 'Product_Type'])['Quantity'].sum().reset_index()
        
            if not seller_product_counts.empty:
                # Create stacked vertical column chart
                fig = px.bar(
                    seller_product_counts,
                    x='Seller_Name',
                    y='Quantity',
                    color='Product_Type',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
        
                fig.update_layout(
                    height=450,  # Bigger chart
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis_title="Seller",
                    yaxis_title="Product Count",
                    legend_title="Product Type",
                    barmode='stack',
                    xaxis=dict(
                        categoryorder='total descending'
                    ),
                    font=dict(color="white", size=14),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255, 255, 255, 0.2)'
                    )
                )
        
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters.")
