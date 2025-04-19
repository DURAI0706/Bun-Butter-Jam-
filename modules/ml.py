import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import matplotlib.dates as mdates
import holidays
import calendar
from dateutil import parser
from mlxtend.frequent_patterns import apriori, association_rules
import os
import numpy as np
# THIS MUST BE THE FIRST STREAMLIT COMMAND

# Load data function with error handling
def convert_to_datetime(df):
    """Convert object columns to datetime with explicit formats"""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                for fmt in [
                    '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d', 
                    '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M %p'
                ]:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt, errors='raise')
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            st.sidebar.success(f"Converted '{col}' to datetime using format: {fmt}")
                            break
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to datetime: {str(e)}")
    return df
    
@st.cache_data
def load_data(uploaded_file=None):
    """Load data from uploaded file or default CSV with caching"""
    try:
        if uploaded_file is not None:
            # Handle user uploaded files
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Load default data file from specific path
            file_path = os.path.join('data', 'Coronation Bakery Dataset.csv')
            if not os.path.exists(file_path):
                st.error(f"Data file not found at: {file_path}")
                st.info("Please ensure your CSV file is in the 'data' directory and named 'Coronation Bakery Dataset.csv'")
                return None
            df = pd.read_csv(file_path)

        # Convert object columns to datetime where applicable
        df = convert_to_datetime(df)

        # Store in session state
        st.session_state['sales_data'] = df
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sales forecasting page with date handling
def sales_forecasting(df):
    st.title("🔮 Product Sales Forecasting")
    
    # User input for month and year
    user_input = st.text_input("Enter month and year (e.g., May 2025)", "May 2025").strip()
    
    try:
        parsed_date = parser.parse("01 " + user_input)
        forecast_start = pd.to_datetime(parsed_date.replace(day=1))
        last_day = calendar.monthrange(parsed_date.year, parsed_date.month)[1]
        forecast_end = pd.to_datetime(parsed_date.replace(day=last_day))
        
        # Ensure 'Date' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        product_types = df['Product_Type'].unique()
        selected_product = st.selectbox("Select Product", product_types)
        
        if st.button("Generate Forecast"):
            st.write(f"\n🔮 Forecasting for {selected_product} - {user_input}")
            prod_df = df[df['Product_Type'] == selected_product].copy()

            daily_sales = prod_df.groupby('Date')['Quantity'].sum().reset_index()
            daily_sales.columns = ['ds', 'y']
            daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])

            if len(daily_sales) < 10:
                st.warning("Not enough data to forecast for this product. Please select another.")
            else:
                with st.spinner('Training forecasting model...'):
                    model = Prophet(
                        yearly_seasonality=False,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                    )
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    model.fit(daily_sales)

                    days_needed = (forecast_end - daily_sales['ds'].max()).days + 1
                    future = model.make_future_dataframe(periods=days_needed)
                    forecast = model.predict(future)

                    month_forecast = forecast[(forecast['ds'] >= forecast_start) & 
                                            (forecast['ds'] <= forecast_end)]

                    if month_forecast.empty:
                        st.warning("Not enough history to reliably predict for this period.")
                    else:
                        # Plot forecast
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.plot(month_forecast['ds'], month_forecast['yhat'], 
                                marker='o', linestyle='-', color='tab:blue', 
                                label='Forecasted Sales')

                        peak_day = month_forecast.loc[month_forecast['yhat'].idxmax()]
                        ax1.axvline(peak_day['ds'], color='red', linestyle='--', alpha=0.5)
                        ax1.text(peak_day['ds'], peak_day['yhat'] + 1, 
                                f"Peak: {int(peak_day['yhat'])} on {peak_day['ds'].date()}", 
                                color='red')

                        ax1.set_title(f"Forecast for {selected_product}")
                        ax1.set_xlabel("Date")
                        ax1.set_ylabel("Expected Units Sold")
                        ax1.grid(True)
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                        plt.xticks(rotation=45)
                        ax1.legend()
                        plt.tight_layout()
                        st.pyplot(fig1)

                        # Plot components
                        st.subheader("Forecast Components")
                        fig2 = model.plot_components(forecast)
                        plt.suptitle(f"Forecast Components for {selected_product}", fontsize=14)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
    except Exception as e:
        st.error(f"Error: {str(e)}. Please enter a valid month and year (e.g., May 2025)")

# Completely rewritten Market basket analysis function
def market_basket_analysis(df):
    st.title("🛒 Market Basket Analysis")
    
    st.write("This analysis identifies products that are frequently purchased together.")
    
    # Check if required columns exist
    required_columns = ['Date', 'Seller_Name', 'Product_Type', 'Quantity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Make a copy to avoid modifying the original dataframe
    basket_df = df.copy()
    
    # Ensure Date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(basket_df['Date']):
        basket_df['Date'] = pd.to_datetime(basket_df['Date'])
    
    # Options for transaction grouping
    st.subheader("Transaction Grouping Settings")
    
    grouping_method = st.radio(
        "Select transaction grouping method:",
        ["Time Window", "Sale Batches", "Random Grouping (Demo)"]
    )
    
    if grouping_method == "Time Window":
        window_size = st.slider(
            "Time window size (in hours):", 
            min_value=1, 
            max_value=24, 
            value=4,
            help="Sales within this time window will be considered part of the same transaction"
        )
        # Create hour component (if not already in the data)
        # In a real dataset, you'd have actual timestamps
        # Here we'll simulate by assigning random hours if needed
        if 'hour' not in basket_df.columns:
            st.info("No hour information found. Simulating time data for demonstration.")
            np.random.seed(42)  # For reproducibility
            basket_df['hour'] = np.random.randint(0, 24, size=len(basket_df))
        
        # Group by date, hour window, and seller
        basket_df['hour_window'] = basket_df['hour'] // window_size
        basket_df['Transaction_ID'] = (basket_df['Date'].dt.strftime('%Y%m%d') + 
                                      '_' + basket_df['hour_window'].astype(str) + 
                                      '_' + basket_df['Seller_Name'])
    
    elif grouping_method == "Sale Batches":
        # Sort by Date and Seller
        basket_df = basket_df.sort_values(['Date', 'Seller_Name'])
        
        # Define when a new transaction might start
        # Either it's a different day or a different seller
        basket_df['New_Group'] = (
            (basket_df['Date'] != basket_df['Date'].shift(1)) | 
            (basket_df['Seller_Name'] != basket_df['Seller_Name'].shift(1))
        )
        
        # Add additional grouping within same day and seller
        # This helps avoid having unrealistically large transactions
        max_items = st.slider(
            "Maximum items per transaction:", 
            min_value=2, 
            max_value=20, 
            value=5,
            help="Split large transactions into smaller ones with at most this many different products"
        )
        
        # Count items within each initial group and create subgroups if needed
        group_id = 0
        transaction_ids = []
        
        current_date = None
        current_seller = None
        current_items = 0
        
        for idx, row in basket_df.iterrows():
            if row['New_Group'] or current_items >= max_items:
                group_id += 1
                current_items = 1
            else:
                current_items += 1
            
            transaction_ids.append(f"Trans_{group_id}")
            
            current_date = row['Date']
            current_seller = row['Seller_Name']
        
        basket_df['Transaction_ID'] = transaction_ids
        
    else:  # Random Grouping
        # This is for demonstration/testing only
        # It creates more random associations between products
        num_transactions = st.slider(
            "Number of synthetic transactions:", 
            min_value=50, 
            max_value=1000, 
            value=200
        )
        
        np.random.seed(42)  # For reproducibility
        basket_df['Transaction_ID'] = np.random.randint(1, num_transactions + 1, size=len(basket_df))
        basket_df['Transaction_ID'] = 'Trans_' + basket_df['Transaction_ID'].astype(str)
    
    # Display sample of transactions
    st.subheader("Sample Transactions")
    sample_size = min(5, len(basket_df['Transaction_ID'].unique()))
    sample_transactions = np.random.choice(basket_df['Transaction_ID'].unique(), sample_size, replace=False)
    
    for trans_id in sample_transactions:
        products = basket_df[basket_df['Transaction_ID'] == trans_id]['Product_Type'].unique()
        st.write(f"**{trans_id}**: {', '.join(products)}")
    
    # Create transaction dataframe for market basket analysis
    @st.cache_data
    def prepare_basket_data(_df):
        try:
            # Create transaction dataframe
            basket = (_df.groupby(['Transaction_ID', 'Product_Type'])['Quantity']
                     .sum()
                     .unstack()
                     .reset_index()
                     .fillna(0)
                     .set_index('Transaction_ID'))
            
            # Convert to binary (1 if product was purchased, 0 otherwise)
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            
            # Basic validation - remove transactions with only one product
            # They don't contribute to association rules
            item_count = basket_sets.sum(axis=1)
            valid_transactions = basket_sets[item_count > 1]
            
            if len(valid_transactions) < 10:
                st.warning(f"Only {len(valid_transactions)} valid transactions found. Results may not be reliable.")
                if len(valid_transactions) == 0:
                    st.error("No valid transactions for analysis. Try different grouping settings.")
                    return None
            
            return valid_transactions
            
        except Exception as e:
            st.error(f"Error preparing basket data: {str(e)}")
            return None
    
    try:
        # Filter data if needed
        filter_data = st.checkbox("Filter data for analysis", False)
        
        if filter_data:
            # Date range filter
            min_date = basket_df['Date'].min().date()
            max_date = basket_df['Date'].max().date()
            
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                basket_df = basket_df[(basket_df['Date'].dt.date >= start_date) & 
                                     (basket_df['Date'].dt.date <= end_date)]
            
            # Product filter
            all_products = sorted(basket_df['Product_Type'].unique())
            selected_products = st.multiselect(
                "Filter specific products (leave empty for all)",
                options=all_products,
                default=[]
            )
            
            if selected_products:
                basket_df = basket_df[basket_df['Product_Type'].isin(selected_products)]
        
        # Create basket data after any filtering
        basket_sets = prepare_basket_data(basket_df)
        
        if basket_sets is not None:
            # Statistics about the dataset
            st.subheader("Dataset Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transactions", len(basket_sets))
            with col2:
                st.metric("Unique Products", basket_sets.shape[1])
            with col3:
                st.metric("Average Products per Transaction", round(basket_sets.sum(axis=1).mean(), 1))
            
            # User inputs for MBA parameters
            st.subheader("Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01, 
                                        help="Products must appear in at least this fraction of transactions")
            with col2:
                min_threshold = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05,
                                         help="Minimum probability that items in consequent appear when items in antecedent are present")
            
            # Additional metrics
            col1, col2 = st.columns(2)
            with col1:
                min_lift = st.slider("Minimum Lift", 1.0, 10.0, 1.2, 0.1,
                                    help="Minimum lift value (how much more likely the consequent is when the antecedent is present)")
            with col2:
                max_length = st.slider("Maximum Items per Rule", 2, 5, 3, 1,
                                      help="Maximum number of items in a rule (antecedent + consequent)")
            
            if st.button("Run Market Basket Analysis"):
                if len(basket_sets) < 10:
                    st.error("Not enough transactions for reliable analysis. Try different grouping settings.")
                else:
                    with st.spinner('Finding frequent itemsets...'):
                        # Find frequent itemsets with progress feedback
                        st.text("Step 1/3: Identifying frequently purchased products...")
                        
                        # Use max_len parameter to control rule complexity
                        frequent_itemsets = apriori(basket_sets, 
                                                   min_support=min_support, 
                                                   use_colnames=True,
                                                   max_len=max_length)
                        
                        if frequent_itemsets.empty:
                            st.warning("No frequent itemsets found with current support threshold. Try lowering the minimum support.")
                        else:
                            st.success(f"Found {len(frequent_itemsets)} frequent itemsets.")
                            
                            # Generate association rules
                            st.text("Step 2/3: Discovering purchase patterns...")
                            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
                            
                            # Filter by lift
                            rules = rules[rules['lift'] >= min_lift]
                            
                            if rules.empty:
                                st.warning("No rules found with current settings. Try adjusting the parameters.")
                            else:
                                # Filter and sort rules
                                st.text("Step 3/3: Analyzing strength of relationships...")
                                rules = rules.sort_values(['confidence', 'lift', 'support'], ascending=[False, False, False])
                                
                                # Display top rules
                                st.subheader("Top Association Rules")
                                
                                # Make the antecedents and consequents more readable
                                rules_display = rules.copy()
                                
                                # Convert frozensets to readable strings
                                rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                                rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                                
                                # Format metrics for better readability
                                pd.options.display.float_format = '{:.1%}'.format
                                rules_display_formatted = rules_display.copy()
                                rules_display_formatted['support'] = rules_display['support'].map('{:.1%}'.format)
                                rules_display_formatted['confidence'] = rules_display['confidence'].map('{:.1%}'.format)
                                rules_display_formatted['lift'] = rules_display['lift'].map('{:.2f}'.format)
                                
                                # Show top rules
                                max_rules = min(10, len(rules_display_formatted))
                                st.dataframe(rules_display_formatted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(max_rules))
                                
                                # Visualize the rules
                                st.subheader("Rule Visualization")
                                
                                if len(rules) >= 3:  # Only plot if we have enough rules
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    scatter = sns.scatterplot(
                                        data=rules.head(20),
                                        x="support",
                                        y="confidence",
                                        size="lift",
                                        hue="lift",
                                        sizes=(50, 500),
                                        ax=ax
                                    )
                                    plt.title("Association Rules (Size/Color = Lift)")
                                    plt.xlabel("Support")
                                    plt.ylabel("Confidence")
                                    
                                    # Add annotations for top rules
                                    for i, row in rules.head(5).iterrows():
                                        ax.text(
                                            row['support'] + 0.01, 
                                            row['confidence'] - 0.02,
                                            f"{', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}", 
                                            fontsize=8
                                        )
                                    
                                    st.pyplot(fig)
                                    
                                    # Network visualization for top rules
                                    st.subheader("Product Network")
                                    st.info("This visualization shows how products are connected through purchase patterns.")
                                    
                                    try:
                                        import networkx as nx
                                        
                                        # Create a graph
                                        G = nx.DiGraph()
                                        
                                        # Add edges from rules
                                        for i, row in rules.head(10).iterrows():
                                            for item_from in row['antecedents']:
                                                for item_to in row['consequents']:
                                                    G.add_edge(item_from, item_to, 
                                                              weight=row['lift'],
                                                              confidence=row['confidence'])
                                        
                                        # Plot the graph
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        pos = nx.spring_layout(G, k=0.5, seed=42)
                                        
                                        # Get edge weights for width
                                        edge_width = [G[u][v]['weight'] for u, v in G.edges()]
                                        
                                        # Draw the graph
                                        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', ax=ax)
                                        nx.draw_networkx_edges(G, pos, width=[w/2 for w in edge_width], 
                                                             edge_color='gray', alpha=0.7,
                                                             connectionstyle='arc3,rad=0.1', ax=ax)
                                        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
                                        
                                        # Add edge labels (confidence)
                                        edge_labels = {(u, v): f"{G[u][v]['confidence']:.0%}" 
                                                     for u, v in G.edges()}
                                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                                                  font_size=8, ax=ax)
                                        
                                        plt.title("Product Relationship Network")
                                        plt.axis('off')
                                        st.pyplot(fig)
                                    except Exception as e:
                                        st.warning(f"Could not create network visualization: {e}")
                                        st.info("To enable network visualization, install networkx library.")
                                
                                # Display actionable insights
                                st.subheader("Actionable Insights")
                                
                                if not rules.empty:
                                    top_rule = rules.iloc[0]
                                    st.write(f"💡 When customers buy **{', '.join(list(top_rule['antecedents']))}**, "
                                            f"they are **{top_rule['confidence']:.0%}** likely to also buy "
                                            f"**{', '.join(list(top_rule['consequents']))}** (lift: {top_rule['lift']:.2f})")
                                    
                                    # More detailed insights
                                    st.write("""
                                    ### Marketing Recommendations:
                                    """)
                                    
                                    recommendations = [
                                        "**Product Placement**: Place frequently co-purchased items near each other in the store",
                                        "**Bundle Offers**: Create discounted bundles for products that are often bought together",
                                        "**Cross-selling**: Train staff to suggest complementary products based on these patterns",
                                        "**Targeted Promotions**: Create promotions that highlight these product combinations"
                                    ]
                                    
                                    for rec in recommendations:
                                        st.write(f"- {rec}")
                                    
                                    # Export option
                                    if st.button("Download Results as CSV"):
                                        csv = rules_display_formatted.to_csv(index=False)
                                        st.download_button(
                                            label="Download CSV",
                                            data=csv,
                                            file_name="market_basket_analysis.csv",
                                            mime="text/csv"
                                        )
    except Exception as e:
        st.error(f"Error in Market Basket Analysis: {str(e)}")
        st.info("Detailed error information for debugging:")
        st.exception(e)

# Main app function
def main():
    st.set_page_config(page_title="Bakery Analytics Dashboard", layout="wide")
    
    st.sidebar.title("Bakery Analytics Dashboard")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Display dataset information
        st.sidebar.subheader("Dataset Information")
        st.sidebar.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Show available columns
        with st.sidebar.expander("Available Columns"):
            st.write(", ".join(df.columns.tolist()))
        
        # Data preview
        with st.sidebar.expander("Data Preview"):
            st.dataframe(df.head(5))
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Sales Forecasting", "Market Basket Analysis"])
        
        if page == "Sales Forecasting":
            sales_forecasting(df)
        elif page == "Market Basket Analysis":
            market_basket_analysis(df)
    else:
        st.error("Please upload a valid dataset or fix the data loading issues.")

if __name__ == "__main__":
    main()
