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
    user_input = st.text_input("Enter month and year (e.g., Jan 2025)", "Jan 2025").strip()
    
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
        st.error(f"Error: {str(e)}. Please enter a valid month and year (e.g., Jan 2025)")

# Completely rewritten Market basket analysis function
def market_basket_analysis(df):
    st.title("🛒 Market Basket Analysis")
    st.write("Select a product to discover what is frequently purchased with it.")

    @st.cache_data
    def prepare_basket_data(_df):
        required_columns = ['Date', 'Seller_Name', 'Product_Type', 'Quantity']
        missing = [col for col in required_columns if col not in _df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        basket_df = _df.copy()
        basket_df['Date'] = pd.to_datetime(basket_df['Date'])
        basket_df['Transaction_ID'] = (
            basket_df['Date'].dt.strftime('%Y%m%d%H%M%S') + '_' + basket_df['Seller_Name']
        )

        product_matrix = (
            basket_df.groupby(['Transaction_ID', 'Product_Type'])['Quantity']
            .sum().unstack().fillna(0)
        )
        basket_sets = product_matrix.applymap(lambda x: 1 if x > 0 else 0)
        return basket_sets, basket_df

    try:
        st.info("Preparing transaction data...")
        basket_sets, original_df = prepare_basket_data(df)

        # Let user select a product
        all_products = basket_sets.columns.tolist()
        selected_product = st.selectbox("Select a Product", sorted(all_products))

        # Auto-run when product is selected
        with st.spinner("Finding product associations..."):
            # Get basic stats about the selected product
            product_count = basket_sets[selected_product].sum()
            total_transactions = len(basket_sets)
            product_frequency = product_count / total_transactions
            
            st.caption(f"Product stats: {selected_product} appears in {product_count} out of {total_transactions} transactions ({product_frequency:.1%})")
            
            # APPROACH 1: Simple co-occurrence analysis (more robust than apriori for small datasets)
            st.subheader(f"Product Associations for: {selected_product}")
            
            # Find co-occurrence with other products
            product_associations = []
            
            # Get transactions containing the selected product
            transactions_with_product = basket_sets[basket_sets[selected_product] == 1]
            
            if len(transactions_with_product) == 0:
                st.warning(f"No transactions found containing {selected_product}")
                return
                
            # Calculate association metrics for each other product
            for other_product in all_products:
                if other_product == selected_product:
                    continue
                
                # Get counts
                both_count = (transactions_with_product[other_product] == 1).sum()
                other_product_count = basket_sets[other_product].sum()
                
                # Skip if no co-occurrences
                if both_count == 0:
                    continue
                    
                # Calculate metrics
                support = both_count / total_transactions
                confidence = both_count / product_count
                other_product_frequency = other_product_count / total_transactions
                lift = confidence / other_product_frequency if other_product_frequency > 0 else 0
                
                # Store the association
                product_associations.append({
                    'other_product': other_product,
                    'both_count': both_count,
                    'support': support,
                    'confidence': confidence,
                    'lift': lift
                })
            
            # Sort by lift and get top associations
            if not product_associations:
                st.warning(f"No meaningful associations found for {selected_product}")
                return
                
            associations_df = pd.DataFrame(product_associations)
            
            # Apply quality filters to prevent overfitting
            # 1. Minimum occurrence filter (prevent rare coincidences from dominating)
            min_occurrence = max(3, int(product_count * 0.05))  # At least 3 co-occurrences or 5% of product's transactions
            filtered_associations = associations_df[associations_df['both_count'] >= min_occurrence]
            
            if filtered_associations.empty:
                # If filter is too strict, take top 5 by raw count
                filtered_associations = associations_df.sort_values('both_count', ascending=False).head(5)
                st.caption("Note: Limited data available. Showing associations based on available transactions.")
            
            # Calculate a balanced score that doesn't overly favor high confidence
            filtered_associations['balanced_score'] = (
                filtered_associations['lift'] * 
                np.sqrt(filtered_associations['support']) *  # Square root gives more weight to support
                np.log1p(filtered_associations['both_count'])  # Log of count favors more frequent associations
            )
            
            # Get top associations by balanced score
            top_associations = filtered_associations.sort_values('balanced_score', ascending=False).head(10)
            
            # Display in a more interpretable format
            display_df = top_associations.copy()
            display_df['support'] = display_df['support'].map('{:.1%}'.format) 
            display_df['confidence'] = display_df['confidence'].map('{:.1%}'.format)
            display_df['lift'] = display_df['lift'].map('{:.2f}'.format)
            
            st.dataframe(
                display_df[['other_product', 'both_count', 'support', 'confidence', 'lift']]
                .rename(columns={
                    'other_product': 'Associated Product',
                    'both_count': 'Co-occurrences',
                    'support': 'Support',
                    'confidence': 'Confidence',
                    'lift': 'Lift'
                }),
                hide_index=True
            )
            
            # Visualization - now based on real co-occurrence stats
            st.subheader("Association Strength Map")
            
            # Create a more informative visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use count for point size to emphasize statistical significance
            scatter = sns.scatterplot(
                data=top_associations,
                x="support", 
                y="confidence",
                size="both_count",  # Size by actual count, not just lift
                hue="lift",
                sizes=(50, 300),
                ax=ax, 
                palette="viridis",
                alpha=0.8
            )
            
            # Add labels for important points
            for i, (_, row) in enumerate(top_associations.head(5).iterrows()):
                ax.text(
                    row['support'] + 0.001, 
                    row['confidence'] + 0.01, 
                    row['other_product'], 
                    fontsize=9, 
                    ha='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )
            
            plt.title(f"Product Association Map for {selected_product}")
            plt.xlabel("Support (% of All Transactions)")
            plt.ylabel("Confidence (% of Selected Product's Transactions)")
            ax.grid(True, linestyle='--', alpha=0.4)
            
            # Set axes to start from 0
            ax.set_xlim(0, max(top_associations['support']) * 1.1)
            ax.set_ylim(0, max(top_associations['confidence']) * 1.1)
            
            st.pyplot(fig)
            
            # Network visualization (alternative view)
            st.subheader("Product Network Visualization")
            
            # Create a network graph with top associations only
            network_df = top_associations.head(7).copy()  # Limit to avoid cluttered graph
            
            G = nx.Graph()
            G.add_node(selected_product, size=20, color='red')
            
            for _, row in network_df.iterrows():
                other_product = row['other_product']
                G.add_node(other_product, size=10, color='blue')
                G.add_edge(selected_product, other_product, weight=row['lift'], count=row['both_count'])
            
            # Draw network graph
            network_fig, network_ax = plt.subplots(figsize=(10, 8))
            
            # Position nodes
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[selected_product], 
                node_size=1200, 
                node_color='orangered', 
                alpha=0.9,
                label="Selected Product"
            )
            
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[n for n in G.nodes() if n != selected_product], 
                node_size=800, 
                node_color='royalblue', 
                alpha=0.7,
                label="Associated Products"
            )
            
            # Draw edges with varying widths based on lift
            edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray')
            
            # Add labels with appropriate font size
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # Add legend
            network_ax.legend()
            
            plt.title(f"Product Association Network for {selected_product}")
            plt.axis('off')
            st.pyplot(network_fig)
            
            # Actionable insights section
            st.subheader("Actionable Insights")
            
            # Get diverse insights
            top_assoc = top_associations.iloc[0]
            second_assoc = top_associations.iloc[1] if len(top_associations) > 1 else None
            
            st.write(f"💡 **Primary Association:** When customers purchase **{selected_product}**, "
                     f"they buy **{top_assoc['other_product']}** in **{top_assoc['confidence']:.0%}** of cases. "
                     f"This association is **{top_assoc['lift']:.1f}x** stronger than random chance.")
            
            if second_assoc is not None:
                st.write(f"💡 **Secondary Association:** Another strong association is with **{second_assoc['other_product']}**, "
                         f"purchased together in **{second_assoc['both_count']}** transactions "
                         f"({second_assoc['confidence']:.0%} of {selected_product} sales).")
            
            # Provide recommendations based on data patterns
            st.markdown("""
            **Recommendation Strategies:**
            - **Cross-Promotion**: Feature associated products on the same promotional materials
            - **Store Placement**: Position these products strategically near each other
            - **Bundle Discounts**: Offer slight discounts when purchased together
            - **Recommendation Engine**: Add these associations to your "Frequently Bought Together" suggestions
            """)
            
    except Exception as e:
        st.error("An error occurred during analysis.")
        st.exception(e)

# Main app function
def main():
    
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
