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
            # First, let's check how common this product is
            product_frequency = basket_sets[selected_product].mean()
            st.caption(f"Product frequency: {selected_product} appears in {product_frequency:.1%} of transactions")
            
            # Dynamically set parameters based on product frequency
            # Less common products need lower support thresholds
            if product_frequency < 0.01:  # Very rare products
                support_values = [0.001, 0.002, 0.005, 0.01]
                max_rules_to_find = 25  # Look for more potential rules for rare items
            elif product_frequency < 0.05:  # Uncommon products
                support_values = [0.005, 0.01, 0.02, 0.03]
                max_rules_to_find = 20
            elif product_frequency < 0.1:  # Moderately common products
                support_values = [0.01, 0.02, 0.03, 0.05]
                max_rules_to_find = 15
            else:  # Common products
                support_values = [0.02, 0.03, 0.05, 0.08]
                max_rules_to_find = 10
                
            # Start with modest confidence to avoid overfitting
            confidence_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
            
            # Prevent overfitting by using diversity metrics
            all_filtered_rules = pd.DataFrame()
                
            # Find a diverse set of rules
            for min_support in support_values:
                if not all_filtered_rules.empty and len(all_filtered_rules) >= max_rules_to_find:
                    break
                    
                frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
                if frequent_itemsets.empty:
                    continue
                    
                for min_conf in confidence_values:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                    if rules.empty:
                        continue
                        
                    # Filter rules containing the selected product
                    filtered_rules = rules[
                        rules['antecedents'].apply(lambda x: selected_product in x) |
                        rules['consequents'].apply(lambda x: selected_product in x)
                    ]
                    
                    if not filtered_rules.empty:
                        # Add these rules to our collection
                        all_filtered_rules = pd.concat([all_filtered_rules, filtered_rules])
                        # Break if we have enough rules
                        if len(all_filtered_rules) >= max_rules_to_find:
                            break
            
            # Process collected rules
            if all_filtered_rules.empty:
                st.warning(f"No meaningful associations found for **{selected_product}**. Try selecting another product.")
                return
                
            # Show parameters used for transparency
            st.caption(f"Analysis parameters dynamically set based on product frequency ({product_frequency:.1%})")
            
            # Anti-overfitting: Apply diversity filters
            # 1. Remove duplicate rules (same antecedents and consequents)
            all_filtered_rules = all_filtered_rules.drop_duplicates(subset=['antecedents', 'consequents'])
            
            # 2. Prioritize rules with higher lift but reasonable confidence
            # This helps avoid 100% confidence rules that might be statistical flukes
            all_filtered_rules['quality_score'] = (
                all_filtered_rules['lift'] * 
                np.log1p(all_filtered_rules['support'] * 100) *  # Log-scale support is more meaningful
                np.tanh(all_filtered_rules['confidence'] * 2)    # Tanh reduces extreme confidence bias
            )
            
            best_rules = all_filtered_rules.sort_values('quality_score', ascending=False).head(15)
            
            # Calculate diversity metrics for display
            rule_count = len(best_rules)
            avg_confidence = best_rules['confidence'].mean()
            avg_lift = best_rules['lift'].mean()
            distinct_products = set()
            
            for idx, row in best_rules.iterrows():
                for item in list(row['antecedents']) + list(row['consequents']):
                    distinct_products.add(item)
            
            st.subheader(f"Top Product Associations for: {selected_product}")
            st.caption(f"Found {rule_count} rules connecting {selected_product} with {len(distinct_products)-1} other products")
            
            # Format for display
            display_rules = best_rules.head(5).copy()  # Only show top 5 to avoid overwhelming
            display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
            display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
            display_rules['support'] = display_rules['support'].map('{:.1%}'.format)
            display_rules['confidence'] = display_rules['confidence'].map('{:.1%}'.format)
            display_rules['lift'] = display_rules['lift'].map('{:.2f}'.format)

            # Remove quality_score before displaying
            st.dataframe(display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

            # Visualization - now with diversity focus
            st.subheader("Association Strength Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Make plot more informative
            scatter = sns.scatterplot(
                data=best_rules,
                x="support", 
                y="confidence",
                size="lift",
                hue="lift",
                sizes=(50, 300),
                ax=ax, 
                palette="viridis",
                alpha=0.7
            )
            
            # Add rule numbers for easier reference
            for i, (idx, row) in enumerate(best_rules.iterrows(), 1):
                if i <= 10:  # Only number the top 10 points
                    ax.text(row['support']+0.001, row['confidence']+0.01, str(i), 
                            fontsize=9, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            plt.title(f"Product Association Map for {selected_product}")
            plt.xlabel("Support (Frequency of Occurrence)")
            plt.ylabel("Confidence (Strength of Association)")
            ax.grid(True, linestyle='--', alpha=0.4)
            
            # Set axes to start from 0
            ax.set_xlim(0, max(best_rules['support']) * 1.1)
            ax.set_ylim(0, max(1.0, max(best_rules['confidence']) * 1.1))
            
            st.pyplot(fig)

            # Actionable insights section - focus on diversity
            st.subheader("Actionable Insights")
            
            # Get multiple top rules
            diverse_insights = []
            processed_items = set([selected_product])
            
            # Get diverse insights by selecting rules with new products
            for _, rule in best_rules.iterrows():
                if len(diverse_insights) >= 3:  # Limit to 3 insights
                    break
                    
                antecedent_items = set(rule['antecedents'])
                consequent_items = set(rule['consequents'])
                
                # Check if this rule introduces new products
                all_items = antecedent_items.union(consequent_items)
                if len(all_items.difference(processed_items)) > 0:
                    # This rule has new items
                    processed_items.update(all_items)
                    diverse_insights.append(rule)
            
            # Show diverse insights
            for i, rule in enumerate(diverse_insights, 1):
                antecedent = ', '.join(sorted(list(rule['antecedents'])))
                consequent = ', '.join(sorted(list(rule['consequents'])))
                
                if selected_product in rule['antecedents']:
                    st.write(f"{i}. When customers buy **{selected_product}**, "
                             f"they are **{rule['confidence']:.1%}** likely to also buy "
                             f"**{consequent}** (lift: {rule['lift']:.2f})")
                else:
                    st.write(f"{i}. When customers buy **{antecedent}**, "
                             f"they are **{rule['confidence']:.1%}** likely to also buy "
                             f"**{selected_product}** (lift: {rule['lift']:.2f})")
            
            # Product recommendation strategies
            st.markdown("""
            **Recommendation Strategies:**
            - **Store Layout**: Position frequently associated products close to each other
            - **Bundle Pricing**: Create discounts for purchasing associated products together
            - **Targeted Marketing**: Show complementary product ads to customers based on purchase history
            - **Personalized Recommendations**: Build "Customers Also Bought" features
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
