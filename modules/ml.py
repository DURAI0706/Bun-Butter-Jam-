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

# Modified Market basket analysis function to work with available columns
def market_basket_analysis(df):
    st.title("🛒 Market Basket Analysis")
    
    st.write("This analysis identifies products that are frequently purchased together.")
    
    # Prepare data for market basket analysis with the available columns
    @st.cache_data
    def prepare_basket_data(_df):
        try:
            # Check if required columns exist
            required_columns = ['Date', 'Seller_Name', 'Product_Type', 'Quantity']
            missing_columns = [col for col in required_columns if col not in _df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
                
            # Make a copy to avoid modifying the original dataframe
            basket_df = _df.copy()
            
            # Create synthetic transaction IDs based on Date and Seller_Name
            # This assumes that each seller has unique transactions per day
            st.info("Creating synthetic transaction IDs using Date and Seller_Name")
            
            # Ensure Date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(basket_df['Date']):
                basket_df['Date'] = pd.to_datetime(basket_df['Date'])
                
            # Create a transaction ID using date and seller name
            basket_df['Transaction_ID'] = basket_df['Date'].dt.strftime('%Y%m%d') + '_' + basket_df['Seller_Name']
            
            # Create transaction dataframe - each row is a transaction, each column is a product
            basket = (basket_df.groupby(['Transaction_ID', 'Product_Type'])['Quantity']
                     .sum()
                     .unstack()
                     .reset_index()
                     .fillna(0)
                     .set_index('Transaction_ID'))
            
            # Convert to binary (1 if product was purchased, 0 otherwise)
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            
            return basket_sets
            
        except Exception as e:
            st.error(f"Error preparing basket data: {str(e)}")
            st.stop()
    
    try:
        st.info("Grouping sales by Date and Seller to create transaction groups")
        basket_sets = prepare_basket_data(df)
        
        # User inputs for MBA parameters
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.2, 0.05, 0.01, 
                                    help="Products must appear in at least this fraction of transactions")
        with col2:
            min_threshold = st.slider("Minimum Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
                                     help="Minimum probability that items in consequent appear when items in antecedent are present")
        
        if st.button("Run Market Basket Analysis"):
            with st.spinner('Finding frequent itemsets...'):
                # Find frequent itemsets with progress feedback
                st.text("Step 1/3: Identifying frequently purchased products...")
                frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
                
                if not frequent_itemsets.empty:
                    # Generate association rules
                    st.text("Step 2/3: Discovering purchase patterns...")
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
                    
                    # Filter and sort rules
                    st.text("Step 3/3: Analyzing strength of relationships...")
                    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
                    
                    # Display top rules
                    st.subheader("Top Association Rules")
                    if rules.empty:
                        st.warning("No strong associations found with current settings. Try lowering the support or confidence thresholds.")
                    else:
                        # Make the antecedents and consequents more readable
                        rules_display = rules.copy()
                        
                        # Convert frozensets to readable strings
                        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        # Format metrics for better readability
                        rules_display['support'] = rules_display['support'].map('{:.1%}'.format)
                        rules_display['confidence'] = rules_display['confidence'].map('{:.1%}'.format)
                        rules_display['lift'] = rules_display['lift'].map('{:.2f}'.format)
                        
                        st.dataframe(rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                        
                        # Visualize the rules
                        st.subheader("Rule Visualization")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = sns.scatterplot(
                            data=rules.head(20),
                            x="support",
                            y="confidence",
                            size="lift",
                            hue="lift",
                            sizes=(20, 200),
                            ax=ax
                        )
                        plt.title("Association Rules (Size = Lift)")
                        plt.xlabel("Support")
                        plt.ylabel("Confidence")
                        st.pyplot(fig)
                        
                        # Display actionable insights
                        st.subheader("Actionable Insights")
                        if not rules.empty:
                            top_rule = rules.iloc[0]
                            st.write(f"💡 When customers buy **{', '.join(list(top_rule['antecedents']))}**, "
                                    f"they are **{top_rule['confidence']:.0%}** likely to also buy "
                                    f"**{', '.join(list(top_rule['consequents']))}** (lift: {top_rule['lift']:.2f})")
                            
                            st.write("""
                            **Recommendations:**
                            - Place these products near each other in the store
                            - Create bundle offers for these product combinations
                            - Use these associations in cross-selling recommendations
                            """)
                else:
                    st.warning("No frequent itemsets found with the current support threshold. Try lowering the minimum support.")
    except Exception as e:
        st.error(f"Error in Market Basket Analysis: {str(e)}")
        st.info("Detailed error information for debugging:")
        st.exception(e)

# Main app function
def main():
    df = load_data()
    
    if df is not None:
        # Display dataset information
        st.sidebar.subheader("Dataset Information")
        st.sidebar.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.sidebar.write("Available columns:", ", ".join(df.columns.tolist()))
        
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
