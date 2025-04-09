import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def detect_column_types(df):
    """Detect column types in the dataframe"""
    col_types = {
        'numeric': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'boolean': df.select_dtypes(include=['bool']).columns.tolist()
    }
    return col_types

def load_data(uploaded_file=None):
    """Load data from uploaded file or default file"""
    try:
        if uploaded_file is not None:
            # Load uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel file
                df = pd.read_excel(uploaded_file)
        else:
            # Load default data
            file_path = os.path.join('data', 'Coronation_Bakery_version_3.csv')
            if not os.path.exists(file_path):
                st.error(f"Data file not found at: {file_path}")
                st.info("Please ensure your CSV file is in the 'data' directory and named 'Coronation_Bakery_version_3.csv'")
                return None
            df = pd.read_csv(file_path)
        
        # Convert date columns to datetime format
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        
        for col in date_cols:
            if col != 'Date' and col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_dynamic_filters(df, col_types):
    """Create dynamic filters in sidebar"""
    st.sidebar.header("Filters")
    filtered_df = df.copy()
    
    date_col = df.select_dtypes(include=['datetime64']).columns.tolist()
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
            filtered_df = filtered_df[(filtered_df[selected_date_col] >= start_date) & 
                    (filtered_df[selected_date_col] <= end_date)]
            
    for col in col_types['numeric'][:3]:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        values = st.sidebar.slider(
            f"Filter {col}",
            min_val, max_val, (min_val, max_val)
        )
        filtered_df = filtered_df[(filtered_df[col] >= values[0]) & (filtered_df[col] <= values[1])]
    
    for col in col_types['categorical'][:3]:
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(
            f"Filter {col}",
            options,
            default=options
        )
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
    
    return filtered_df

def create_filters(df):
    """Create filters in main dashboard area with improved state management"""
    # Initialize session state if needed
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = df['Date'].min().date()
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = df['Date'].max().date()
    if 'selected_seller' not in st.session_state:
        st.session_state['selected_seller'] = "All Sellers"
    if 'selected_product' not in st.session_state:
        st.session_state['selected_product'] = "All Products"
    if 'seller_options' not in st.session_state:
        st.session_state['seller_options'] = ["All Sellers"] + sorted(df['Seller_Name'].unique().tolist())
    
    # Define callback functions for filter changes
    def on_seller_change():
        selected_seller = st.session_state.seller_select
        if selected_seller != st.session_state['selected_seller']:
            st.session_state['selected_seller'] = selected_seller
            st.session_state['selected_product'] = "All Products"
            # Update product options based on seller selection
            if selected_seller == "All Sellers":
                product_df = df
            else:
                product_df = df[df['Seller_Name'] == selected_seller]
            st.session_state['product_options'] = ["All Products"] + sorted(product_df['Product_Type'].unique().tolist())
    
    def on_product_change():
        st.session_state['selected_product'] = st.session_state.product_select
    
    def on_start_date_change():
        st.session_state['start_date'] = st.session_state.start_date_input
        
    def on_end_date_change():
        st.session_state['end_date'] = st.session_state.end_date_input
        
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.date_input(
            "Start Date",
            value=st.session_state['start_date'],
            min_value=df['Date'].min().to_pydatetime(),
            max_value=df['Date'].max().to_pydatetime(),
            format="DD-MM-YYYY",
            key="start_date_input",
            on_change=on_start_date_change
        )

    with col2:
        st.date_input(
            "End Date",
            value=st.session_state['end_date'],
            min_value=df['Date'].min().to_pydatetime(),
            max_value=df['Date'].max().to_pydatetime(),
            format="DD-MM-YYYY",
            key="end_date_input",
            on_change=on_end_date_change
        )

        if st.session_state['end_date'] < st.session_state['start_date']:
            st.error("End date must be after start date. Resetting to default range.")
            st.session_state['start_date'] = df['Date'].min().date()
            st.session_state['end_date'] = df['Date'].max().date()

    with col3:
        # Update product options if needed when changing sellers
        if selected_seller := st.selectbox(
            "Select Seller",
            options=st.session_state['seller_options'],
            index=st.session_state['seller_options'].index(st.session_state['selected_seller']),
            key="seller_select",
            on_change=on_seller_change
        ):
            pass  # The on_change callback will handle the state updates

    with col4:
        # Make sure product options are initialized based on current seller
        if 'product_options' not in st.session_state:
            if st.session_state['selected_seller'] == "All Sellers":
                product_df = df
            else:
                product_df = df[df['Seller_Name'] == st.session_state['selected_seller']]
            st.session_state['product_options'] = ["All Products"] + sorted(product_df['Product_Type'].unique().tolist())
        
        # Verify the selected product is in the options
        if st.session_state['selected_product'] not in st.session_state['product_options']:
            st.session_state['selected_product'] = "All Products"
            
        st.selectbox(
            "Select Product Type",
            options=st.session_state['product_options'],
            index=st.session_state['product_options'].index(st.session_state['selected_product']),
            key="product_select",
            on_change=on_product_change
        )

def filter_data(df):
    """Apply filters based on session state"""
    start_date = st.session_state['start_date']
    end_date = st.session_state['end_date']
    selected_seller = st.session_state['selected_seller']
    selected_product = st.session_state['selected_product']
    
    # Filter by date range
    filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    
    if filtered_df.empty:
        st.warning("No data available for the selected date range. Please adjust your filter.")
        return df
        
    # Filter by seller if not "All Sellers"
    if selected_seller != "All Sellers":
        seller_filtered = filtered_df[filtered_df['Seller_Name'] == selected_seller]
        if seller_filtered.empty:
            st.warning(f"No data available for seller '{selected_seller}' in the selected date range.")
            return filtered_df
        filtered_df = seller_filtered
        
    # Filter by product if not "All Products"
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
    
    # Container 4: Product Count by Seller
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
                    font=dict(size=14),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(200, 200, 200, 0.2)'
                    )
                )
        
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters.")

def load_module():
    st.set_page_config(
        page_title="Coronation Bakery Dashboard",
        page_icon="🍰",
        layout="wide"
    )
    
    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file (optional, defaults to Coronation Bakery Dataset)",
        type=["csv", "xlsx", "xls"]
    )

    # Load either the uploaded file or the default dataset
    df = load_data(uploaded_file)
    if df is None:
        st.error("❌ Failed to load dataset.")
        return

    # Set dataset name
    if uploaded_file:
        dataset_name = uploaded_file.name.split('.')[0].replace('_', ' ').title()
        st.sidebar.success(f"✅ Loaded file: {uploaded_file.name}")
    else:
        dataset_name = "Coronation Bakery Dataset"
        st.sidebar.info("📂 Using default: Coronation Bakery Dataset.csv")

    # Detect column types
    col_types = detect_column_types(df)
    
    # Apply dynamic filters from sidebar
    filtered_df = create_dynamic_filters(df, col_types)
    
    # Display main dashboard
    st.markdown(f"<h1 style='text-align: center;'>{dataset_name} Sales Dashboard</h1>", unsafe_allow_html=True)
    
    # Initialize sellers and products list if not already in session state
    if 'sales_data' not in st.session_state:
        st.session_state['sales_data'] = filtered_df
    
    # Display main filters, metrics and charts
    create_filters(filtered_df)
    display_metrics(filtered_df)
    display_charts(filtered_df)
    
    st.sidebar.success(f"✅ {len(filtered_df)} rows × {len(filtered_df.columns)} columns")

if __name__ == "__main__":
    main()
