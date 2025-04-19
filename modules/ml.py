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

# Set page config
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sales Forecasting", "Market Basket Analysis"])

# Load data
df = pd.read_csv("Coronation Bakery Dataset.csv")

if page == "Sales Forecasting":
    st.title("🔮 Product Sales Forecasting")
    
    # User input for month and year
    user_input = st.text_input("Enter month and year (e.g., May 2025)", "May 2025").strip()
    
    try:
        parsed_date = parser.parse("01 " + user_input)
        forecast_start = pd.to_datetime(parsed_date.replace(day=1))
        last_day = calendar.monthrange(parsed_date.year, parsed_date.month)[1]
        forecast_end = pd.to_datetime(parsed_date.replace(day=last_day))
        
        product_types = df['Product_Type'].unique()
        selected_product = st.selectbox("Select Product", product_types)
        
        if st.button("Generate Forecast"):
            st.write(f"\n🔮 Forecasting for {selected_product} - {user_input}")
            prod_df = df[df['Product_Type'] == selected_product]

            daily_sales = prod_df.groupby('Date')['Quantity'].sum().reset_index()
            daily_sales.columns = ['ds', 'y']

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

                    month_forecast = forecast[(forecast['ds'] >= forecast_start) & (forecast['ds'] <= forecast_end)]

                    if month_forecast.empty:
                        st.warning("Not enough history to reliably predict for this period.")
                    else:
                        # Plot forecast
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.plot(month_forecast['ds'], month_forecast['yhat'], marker='o', linestyle='-', color='tab:blue', label='Forecasted Sales')

                        peak_day = month_forecast.loc[month_forecast['yhat'].idxmax()]
                        ax1.axvline(peak_day['ds'], color='red', linestyle='--', alpha=0.5)
                        ax1.text(peak_day['ds'], peak_day['yhat'] + 1, f"Peak: {int(peak_day['yhat'])} on {peak_day['ds'].date()}", color='red')

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
        st.error(f"Error: {e}. Please enter a valid month and year (e.g., May 2025)")

elif page == "Market Basket Analysis":
    st.title("🛒 Market Basket Analysis")
    
    st.write("""
    This analysis identifies products that are frequently purchased together.
    """)
    
    # Prepare data for market basket analysis
    @st.cache_data
    def prepare_basket_data():
        # Create a transaction dataframe (one-hot encoded)
        basket = df.groupby(['Transaction_ID', 'Product_Type'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('Transaction_ID')
        # Convert to binary (1 if product was purchased, 0 otherwise)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        return basket_sets
    
    basket_sets = prepare_basket_data()
    
    # User inputs for MBA parameters
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.2, 0.05, 0.01)
    with col2:
        min_threshold = st.slider("Minimum Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    if st.button("Run Market Basket Analysis"):
        with st.spinner('Finding frequent itemsets...'):
            # Find frequent itemsets
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
            
            if not frequent_itemsets.empty:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
                
                # Filter and sort rules
                rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
                
                # Display top rules
                st.subheader("Top Association Rules")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                
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
                top_rule = rules.iloc[0]
                st.write(f"💡 When customers buy **{', '.join(top_rule['antecedents'])}**, they are **{top_rule['confidence']:.0%}** likely to also buy **{', '.join(top_rule['consequents'])}** (lift: {top_rule['lift']:.2f})")
                
                st.write("""
                **Recommendations:**
                - Place these products near each other in the store
                - Create bundle offers for these product combinations
                - Use these associations in cross-selling recommendations
                """)
            else:
                st.warning("No association rules found with the current parameters. Try lowering the minimum support or confidence threshold.")
