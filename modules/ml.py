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

    @st.cache_data
    def prepare_basket_data(_df):
        required_columns = ['Date', 'Seller_Name', 'Product_Type', 'Quantity']
        missing = [col for col in required_columns if col not in _df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        basket_df = _df.copy()
        basket_df['Date'] = pd.to_datetime(basket_df['Date'])

        # 👇 Better Transaction ID: Includes time to avoid over-grouping
        basket_df['Transaction_ID'] = (
            basket_df['Date'].dt.strftime('%Y%m%d%H%M%S') + '_' + basket_df['Seller_Name']
        )

        # 👇 One-hot encoded product matrix (binary)
        product_matrix = (
            basket_df.groupby(['Transaction_ID', 'Product_Type'])['Quantity']
            .sum().unstack().fillna(0)
        )
        basket_sets = product_matrix.applymap(lambda x: 1 if x > 0 else 0)

        return basket_sets

    try:
        st.info("Preparing transaction data...")
        basket_sets = prepare_basket_data(df)

        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.2, 0.05, 0.01)
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)

        if st.button("Run Market Basket Analysis"):
            with st.spinner("Analyzing basket data..."):
                # 🧮 Frequent itemsets
                frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
                if frequent_itemsets.empty:
                    st.warning("No frequent itemsets found. Try lowering support.")
                    return

                # 🔗 Association rules
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                if rules.empty:
                    st.warning("No association rules found. Try lowering confidence.")
                    return

                # 🎯 Top 5 sorted by confidence, then lift
                rules = rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
                top_rules = rules.head(5).copy()

                # 📋 Display nicely
                st.subheader("Top 5 Association Rules")
                top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
                top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
                top_rules['support'] = top_rules['support'].map('{:.1%}'.format)
                top_rules['confidence'] = top_rules['confidence'].map('{:.1%}'.format)
                top_rules['lift'] = top_rules['lift'].map('{:.2f}'.format)

                st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

                # 📊 Visualization
                st.subheader("Rule Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=rules.head(20),
                    x="support",
                    y="confidence",
                    size="lift",
                    hue="lift",
                    sizes=(40, 200),
                    ax=ax
                )
                plt.title("Association Rules (Size = Lift)")
                plt.xlabel("Support")
                plt.ylabel("Confidence")
                st.pyplot(fig)

                # 💡 Actionable insight
                st.subheader("Actionable Insight")
                top_rule = rules.iloc[0]
                st.write(f"💡 If customers buy **{', '.join(list(top_rule['antecedents']))}**, "
                         f"they are **{top_rule['confidence']:.0%}** likely to also buy "
                         f"**{', '.join(list(top_rule['consequents']))}** (lift: {top_rule['lift']:.2f})")
                st.markdown("""
                **Recommendations:**
                - Cross-promote or bundle these items
                - Highlight these pairs in marketing emails
                - Optimize shelf placement or app suggestions
                """)
    except Exception as e:
        st.error("An error occurred during analysis.")
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
