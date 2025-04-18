import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_extras.dataframe_explorer import dataframe_explorer
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import os

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

def detect_column_types(df):
    """Detect column types and return categorized lists"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    for col in cat_cols[:]:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            cat_cols.remove(col)
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    return {
        'numeric': numeric_cols,
        'categorical': cat_cols,
        'datetime': date_cols,
        'binary': binary_cols
    }

def generate_kpis(df, col_types):
    """Generate dynamic KPIs based on dataset characteristics"""
    kpis = []
    for col in col_types['numeric'][:3]:
        kpis.append({
            'title': f"Total {col}",
            'value': f"{df[col].sum():,.2f}",
            'delta': None
        })
        kpis.append({
            'title': f"Avg {col}",
            'value': f"{df[col].mean():,.2f}",
            'delta': None
        })
    for col in col_types['categorical'][:2]:
        kpis.append({
            'title': f"Unique {col}",
            'value': f"{df[col].nunique():,}",
            'delta': None
        })
    if col_types['datetime']:
        date_col = col_types['datetime'][0]
        date_range = df[date_col].max() - df[date_col].min()
        kpis.append({
            'title': "Date Range",
            'value': f"{date_range.days} days",
            'delta': None
        })
    return kpis

def style_metric_cards():
    st.markdown("""
        <style>
        div.metric-container {
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 8px;
            box-shadow: 0 0 8px rgba(0, 123, 255, 0.3);
            background-color: transparent; /* Changed from #F0F2F6 */
            transition: box-shadow 0.3s ease;
            padding: 10px;
            text-align: center;
            margin-bottom: 10px;
        }

        div.metric-container:hover {
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

def show_kpi_cards(kpis, columns_per_row=3):
    """Display dynamic KPI cards in a matrix layout with individual containers"""
    style_metric_cards()
    
    for i in range(0, len(kpis), columns_per_row):
        row_kpis = kpis[i:i + columns_per_row]
        cols = st.columns(len(row_kpis))
        
        for j, kpi in enumerate(row_kpis):
            with cols[j]:
                # Create a container for each metric
                with st.container():
                    # Add custom HTML with a container div surrounding each metric
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-title">{kpi['title']}</p>
                        <p class="metric-value">{kpi['value']}</p>
                        {f'<p class="metric-delta">{kpi["delta"]}</p>' if kpi['delta'] else ''}
                    </div>
                    """, unsafe_allow_html=True)


def show_missing_values(df):
    """Show missing values analysis"""
    st.subheader("Missing Values Analysis")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        st.success("No missing values found!")
        return
    col1, col2 = st.columns(2)
    with col1:
        missing_df = missing.reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
        st.dataframe(missing_df.sort_values('Missing %', ascending=False))
    with col2:
        fig = px.imshow(df.isna(),
                       color_continuous_scale='gray',
                       title="Missing Values Heatmap (Yellow = Missing)")
        st.plotly_chart(fig, use_container_width=True)

def show_correlations(df, col_types):
    """
    Enhanced correlation analysis with label encoding for categorical columns
    Combines the best features of both versions with error fixes
    """
    if len(col_types['numeric']) < 2 and len(col_types['categorical']) < 2:
        st.warning("Not enough numeric or categorical columns for correlation analysis (need at least 2 columns)")
        return
    st.subheader("📈 Feature Correlations Analysis")
    df_corr = df.copy()
    if len(col_types['categorical']) > 0:
        le = LabelEncoder()
        categorical_cols = col_types['categorical']
        for col in categorical_cols:
            try:
                df_corr[col] = le.fit_transform(df_corr[col].astype(str))
            except Exception as e:
                st.warning(f"Could not encode column '{col}': {str(e)}")
                df_corr.drop(col, axis=1, inplace=True)
                if col in col_types['numeric']:
                    col_types['numeric'].remove(col)
    all_cols = col_types['numeric'] + [col for col in col_types['categorical'] if col in df_corr.columns]
    
    if len(all_cols) < 2:
        st.warning("Not enough valid columns for correlation analysis after encoding")
        return
    corr_method = st.radio("Correlation method", 
                          ['pearson', 'spearman', 'kendall'],
                          horizontal=True,
                          key='corr_method_selector')
    corr = df_corr[all_cols].corr(method=corr_method)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Correlation Ranking")
        corr_unstacked = corr.abs().unstack()
        corr_unstacked = corr_unstacked[corr_unstacked < 1]  # Remove self-correlations
        corr_pairs = corr_unstacked.sort_values(ascending=False).reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        dtypes = df.dtypes.astype(str).to_dict()
        corr_pairs['Type 1'] = corr_pairs['Feature 1'].map(dtypes)
        corr_pairs['Type 2'] = corr_pairs['Feature 2'].map(dtypes)
        corr_pairs['Correlation'] = corr_pairs['Correlation'].round(3)
        corr_pairs = corr_pairs.drop_duplicates(subset=['Correlation'])
        def color_high_correlations(val):
            color = 'red' if abs(val) > 0.7 else 'orange' if abs(val) > 0.5 else 'blue'
            return f'color: {color}'

        display_df = corr_pairs.head(20).copy()
        for col in ['Feature 1', 'Feature 2', 'Type 1', 'Type 2']:
            display_df[col] = display_df[col].astype(str)
        st.dataframe(
            display_df.style.map(color_high_correlations, subset=['Correlation']),
            height=600,
            use_container_width=True
        )
        if len(col_types['categorical']) > 0:
            st.info("ℹ️ Categorical columns were label encoded for correlation analysis")
    with col2:
        st.markdown("### Correlation Matrix")
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title=f"Feature Correlation Heatmap (Method: {corr_method.title()})"
        )
        fig.update_layout(
            margin=dict(l=50, r=50, b=100, t=100),
            xaxis=dict(tickangle=45),
            coloraxis_colorbar=dict(
                title="Correlation",
                thickness=20,
                len=0.75
            )
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("🔍 How to interpret correlations"):
        st.markdown(f"""
        **Interpretation Guide ({corr_method.title()} Correlation)**:
    
        - <span style='color:red;'>**+1 to +0.7**</span>: 🔴 Very strong positive relationship  
        - <span style='color:orange;'>**+0.7 to +0.3**</span>: 🟠 Positive relationship  
        - <span style='color:blue;'>**-0.3 to +0.3**</span>: 🔵 Little to no relationship  
        - <span style='color:orange;'>**-0.3 to -0.7**</span>: 🟠 Negative relationship  
        - <span style='color:red;'>**-0.7 to -1**</span>: 🔴 Very strong negative relationship
    
        **Method Notes**:
        - **Pearson**: Measures linear relationships (for continuous variables)  
        - **Spearman**: Measures monotonic relationships (for ordinal/ranked data)  
        - **Kendall**: Similar to Spearman but more robust for small samples  
    
        **Note for categorical variables**:
        - Correlations with label-encoded categorical variables should be interpreted with caution  
        - The strength depends on how categories are distributed  
        - Consider using other statistical tests for categorical-categorical relationships  
        """, unsafe_allow_html=True)
        
        st.warning("Note: Correlation ≠ Causation. High correlation may indicate a relationship but doesn't prove one causes the other.")


def show_distributions(df, col_types):
    """Show distributions for different column types"""
    st.subheader("Data Distributions")
    if col_types['numeric']:
        st.write("#### Numeric Columns")
        num_col = st.selectbox("Select numeric column", col_types['numeric'])
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y=num_col, title=f"Box Plot of {num_col}")
            st.plotly_chart(fig, use_container_width=True)
            q1 = df[num_col].quantile(0.25)
            q3 = df[num_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[num_col] < lower_bound) | (df[num_col] > upper_bound)]
            st.info(f"Detected {len(outliers)} outliers using IQR method")
    if col_types['categorical']:
        st.write("#### Categorical Columns")
        cat_col = st.selectbox("Select categorical column", col_types['categorical'])
        value_counts = df[cat_col].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(value_counts, x='Value', y='Count', 
                        title=f"Distribution of {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(value_counts, values='Count', names='Value',
                        title=f"Proportion of {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

def show_time_series(df, col_types):
    """Show time series analysis with updated resample codes"""
    if not col_types['datetime']:
        return
    
    st.subheader("Time Series Analysis")
    date_col = st.selectbox("Select date column", col_types['datetime'], key='ts_date_col')
    
    if not col_types['numeric']:
        st.warning("No numeric columns for time series visualization")
        return
    
    value_col = st.selectbox("Select value column", col_types['numeric'], key='ts_value_col')
    resample_options = {
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'ME',
        'Quarterly': 'QE',
        'Yearly': 'YE'
    }
    selected_freq = st.selectbox(
        "Resampling frequency", 
        list(resample_options.keys()),
        key='ts_resample_freq'
    )
    ts_df = df.set_index(date_col)[value_col]\
             .resample(resample_options[selected_freq])\
             .mean()\
             .reset_index()
    
    fig = px.line(ts_df, x=date_col, y=value_col, 
                 title=f"{value_col} over Time")
    st.plotly_chart(fig, use_container_width=True)

def safe_display_dataframe(df, height=None):
    """Helper function to safely display dataframes with consistent types"""
    display_df = df.copy()
    for col in display_df.columns:
        if not (pd.api.types.is_numeric_dtype(display_df[col]) or 
                pd.api.types.is_datetime64_any_dtype(display_df[col])):
            display_df[col] = display_df[col].astype(str)
    
    st.dataframe(display_df, height=height, use_container_width=True)

def create_filters(df, col_types):
    """Create dynamic filters in sidebar"""
    st.sidebar.header("Filters")
    
    # Filter by date
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if date_cols:
        selected_date_col = st.sidebar.selectbox("Select date column for filtering", date_cols)
        min_date = df[selected_date_col].min()
        max_date = df[selected_date_col].max()
        
        # Ensure dates are not NaT
        if pd.isnull(min_date) or pd.isnull(max_date):
            st.warning(f"No valid dates found in '{selected_date_col}' column.")
        else:
            date_range = st.sidebar.date_input(
                f"Select date range for {selected_date_col}",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                df = df[(df[selected_date_col] >= start_date) & (df[selected_date_col] <= end_date)]

    # Filter by numeric columns
    for col in col_types.get('numeric', [])[:3]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            if pd.notnull(min_val) and pd.notnull(max_val):
                values = st.sidebar.slider(
                    f"Filter {col}",
                    min_val, max_val, (min_val, max_val)
                )
                df = df[(df[col] >= values[0]) & (df[col] <= values[1])]

    # Filter by categorical columns
    for col in col_types.get('categorical', [])[:3]:
        if col in df.columns:
            options = df[col].dropna().unique().tolist()
            if options:
                selected = st.sidebar.multiselect(
                    f"Filter {col}",
                    options,
                    default=options
                )
                df = df[df[col].isin(selected)]

    return df

def show_data_preview(df):
    """Show interactive data preview with dynamic features"""
    st.subheader("📊 Data Preview")
    if len(df) > 10000:
        st.warning("Large dataset detected! Showing sample of 10,000 rows.")
        sample_df = df.sample(10000)
    else:
        sample_df = df.copy()
    tab1, tab2, tab3 = st.tabs(["Interactive Explorer", "Quick Stats", "Data Types"])
    with tab1:
        height = min(600, 100 + len(sample_df) * 25)
        filtered_df = dataframe_explorer(sample_df)
        st.dataframe(filtered_df, 
                    use_container_width=True, 
                    height=height)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download filtered data as CSV",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name='filtered_data.csv',
                mime='text/csv'
            )
        with col2:
            st.download_button(
                label="Download full dataset as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='full_data.csv',
                mime='text/csv'
            )
    with tab2:
        st.write("### Descriptive Statistics")
        stat_tabs = st.tabs(["Numerical", "Categorical", "All Columns"]) 
        with stat_tabs[0]:
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                st.dataframe(df[num_cols].describe().T.style.background_gradient(cmap='Blues'),
                            use_container_width=True)
            else:
                st.warning("No numerical columns found")
        with stat_tabs[1]:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                cat_stats = pd.DataFrame({
                    'Unique Values': df[cat_cols].nunique(),
                    'Most Common': df[cat_cols].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'),
                    'Frequency': df[cat_cols].apply(lambda x: x.value_counts().iloc[0] if len(x.value_counts()) > 0 else 0)
                })
                st.dataframe(cat_stats.style.background_gradient(cmap='Greens'),
                            use_container_width=True)
            else:
                st.warning("No categorical columns found")
        with stat_tabs[2]:
            st.dataframe(df.describe(include='all').T.style.background_gradient(cmap='Purples'),
                        use_container_width=True)    
    with tab3:
        st.write("### Data Types Overview")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Missing Values': df.isna().sum(),
            '% Missing': (df.isna().mean() * 100).round(2),
            'Unique Values': df.nunique()
        }).sort_values(by='Type')        
        st.dataframe(dtype_info.style.bar(subset=['% Missing'], color='#ff6961'))
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

def show_correlation_visualizations(df, col_types):
    # Apply styling for metric cards at the beginning of the function
    style_metric_cards()
    
    # Helper function to identify date columns
    def get_date_column(df):
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            return date_cols[0]        
        for col in df.select_dtypes(include=['datetime']).columns:
            return col
        return None
    
    # Dashboard header
    st.markdown("## 📊 Sales Analytics Dashboard")
    
    # Identify key metrics columns
    amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'total' in col.lower()]
    quantity_cols = [col for col in df.columns if 'quantity' in col.lower() or 'qty' in col.lower()]
    sales_metric = amount_cols[0] if amount_cols else (quantity_cols[0] if quantity_cols else None)
       
    # Main dashboard layout
    tab_names = ["Performance", "Trends", "Distributions", "Relationships", "Advanced"]
    main_tabs = st.tabs(tab_names)
    
    # Tab 1: Performance Analysis
    with main_tabs[0]:
        if sales_metric:
            st.subheader("🏆 Top Performers")
            cat_cols = col_types['categorical'] if 'categorical' in col_types else df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                perf_tab_names = [f"By {col}" for col in cat_cols[:3]]
                perf_tabs = st.tabs(perf_tab_names)
                
                for i, col in enumerate(cat_cols[:3]):
                    with perf_tabs[i]:
                        # Create a container for the entire tab content
                        tab_container = st.container()
                        
                        with tab_container:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                # Create a container specifically for the chart
                                chart_container = st.container()
                                with chart_container:
                                    top_items = df.groupby(col)[sales_metric].sum().nlargest(5).reset_index()
                                    fig = px.bar(top_items, x=col, y=sales_metric, 
                                                color=col, title=f"Top {col} by {sales_metric}")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Create a container specifically for the metrics
                                metrics_container = st.container()
                                with metrics_container:
                                    # Apply metric card style to key insight
                                    st.markdown("""
                                    <div class="metric-container">
                                        <p class="metric-title">Key Insight</p>
                                        <p class="metric-value">{}</p>
                                        <p>leads with <b>{:,.2f}</b> in {}</p>
                                    </div>
                                    """.format(top_items.iloc[0][col], top_items.iloc[0][sales_metric], sales_metric), 
                                    unsafe_allow_html=True)
                                    
                                    # Show proportion of top performer
                                    total = df[sales_metric].sum()
                                    proportion = top_items.iloc[0][sales_metric] / total
                                    st.progress(proportion)
                                    st.markdown("""
                                    <div class="metric-container">
                                        <p class="metric-title">Proportion</p>
                                        <p class="metric-value">{:.1%}</p>
                                        <p>of total</p>
                                    </div>
                                    """.format(proportion), unsafe_allow_html=True)
        else:
            st.warning("No sales metric columns found for performance analysis")
                
    # Tab 2: Trend Analysis
    with main_tabs[1]:
        date_col = get_date_column(df)
        if date_col and sales_metric:
            st.subheader("📈 Temporal Trends")
            
            # Create a main container for the entire trend analysis
            trend_container = st.container()
            
            with trend_container:
                trend_col1, trend_col2 = st.columns([1, 3])
                
                with trend_col1:
                    # Container for controls and metrics
                    controls_container = st.container()
                    with controls_container:
                        # Controls for trend analysis
                        metric_options = []
                        if sales_metric: metric_options.append(sales_metric)
                        if quantity_cols: metric_options.extend(quantity_cols[:2])
                        if amount_cols: metric_options.extend(amount_cols[:2])
                        
                        selected_metric = st.selectbox("Select metric", metric_options, key="trend_metric_select")
                        
                        time_groups = st.radio("Time grouping", 
                                             ['Daily', 'Weekly', 'Monthly', 'Quarterly'],
                                             key="time_group_select")
                        
                        st.markdown("### Trend Insight")
                        df[date_col] = pd.to_datetime(df[date_col])
                        
                        # Calculate trend data based on time grouping
                        if time_groups == 'Daily':
                            trend_data = df.groupby(df[date_col].dt.date)[selected_metric].sum().reset_index()
                            x_col = date_col
                        elif time_groups == 'Weekly':
                            trend_data = df.groupby([df[date_col].dt.year.rename("Year"), df[date_col].dt.isocalendar().week.rename("Week")])[selected_metric].sum().reset_index()
                            trend_data['Week'] = trend_data['Year'].astype(str) + '-W' + trend_data['Week'].astype(str).str.zfill(2)
                            x_col = 'Week'
                        elif time_groups == 'Monthly':
                            trend_data = df.groupby([df[date_col].dt.year.rename("Year"), df[date_col].dt.month.rename("Month")])[selected_metric].sum().reset_index()
                            trend_data['Month'] = trend_data['Year'].astype(str) + '-' + trend_data['Month'].astype(str).str.zfill(2)
                            x_col = 'Month'
                        else:
                            trend_data = df.groupby([df[date_col].dt.year.rename("Year"), df[date_col].dt.quarter.rename("Quarter")])[selected_metric].sum().reset_index()
                            trend_data['Quarter'] = trend_data['Year'].astype(str) + '-Q' + trend_data['Quarter'].astype(str)
                            x_col = 'Quarter'
                        
                        peak_val = trend_data[selected_metric].max()
                        peak_time = trend_data.loc[trend_data[selected_metric].idxmax(), x_col]
                        
                        # Apply metric card style to peak value
                        st.markdown("""
                        <div class="metric-container">
                            <p class="metric-title">Peak Period</p>
                            <p class="metric-value">{}</p>
                            <p>with <b>{:,.2f}</b></p>
                        </div>
                        """.format(peak_time, peak_val), unsafe_allow_html=True)
                        
                        # Calculate growth
                        first_val = trend_data.iloc[0][selected_metric]
                        last_val = trend_data.iloc[-1][selected_metric]
                        growth = (last_val - first_val) / first_val if first_val != 0 else 0
                        
                        # Apply metric card style to growth
                        st.markdown("""
                        <div class="metric-container">
                            <p class="metric-title">Period Growth</p>
                            <p class="metric-value">{:.1%}</p>
                            <p>{:.1f}% change</p>
                        </div>
                        """.format(growth, growth*100), unsafe_allow_html=True)
                
                with trend_col2:
                    # Container specifically for the chart
                    chart_container = st.container()
                    with chart_container:
                        fig = px.line(trend_data, x=x_col, y=selected_metric, 
                                     title=f"{time_groups} Trend of {selected_metric}")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date column or sales metric not found for trend analysis")
    
    # Tab 3: Distributions
    with main_tabs[2]:
        st.subheader("📊 Distribution Analysis")
        
        num_cols = col_types['numeric'] if 'numeric' in col_types else df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = col_types['categorical'] if 'categorical' in col_types else df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if num_cols:
            # Create a container for the entire distribution analysis
            dist_container = st.container()
            
            with dist_container:
                # First row - Distribution controls and statistics
                controls_container = st.container()
                with controls_container:
                    dist_col = st.selectbox("Select metric", num_cols)
                    if cat_cols:
                        group_col = st.selectbox("Group by", ['None'] + cat_cols)
                    
                    st.markdown("### Statistics")
                    stats = df[dist_col].describe()
                    st.dataframe(pd.DataFrame(stats).T.style.highlight_max(axis=1, color='#c76b7a'))
                    
                    # Apply metric card style to key statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("""
                        <div class="metric-container">
                            <p class="metric-title">Mean</p>
                            <p class="metric-value">{:,.2f}</p>
                        </div>
                        """.format(stats['mean']), unsafe_allow_html=True)
                    with col2:
                        st.markdown("""
                        <div class="metric-container">
                            <p class="metric-title">Median</p>
                            <p class="metric-value">{:,.2f}</p>
                        </div>
                        """.format(stats['50%']), unsafe_allow_html=True)
                    with col3:
                        st.markdown("""
                        <div class="metric-container">
                            <p class="metric-title">Standard Deviation</p>
                            <p class="metric-value">{:,.2f}</p>
                        </div>
                        """.format(stats['std']), unsafe_allow_html=True)
                
                # Second row - Distribution chart
                chart_container = st.container()
                with chart_container:
                    if cat_cols and group_col != 'None':
                        fig = px.box(df, x=group_col, y=dist_col, color=group_col,
                                    title=f"Distribution of {dist_col} by {group_col}")
                    else:
                        fig = px.histogram(df, x=dist_col, 
                                        title=f"Distribution of {dist_col}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found for distribution analysis")
            
        if cat_cols and sales_metric:
            st.subheader("🍰 Composition Analysis")
            
            # Create a container for composition analysis
            comp_container = st.container()
            
            with comp_container:
                # First composition row - Pie chart
                pie_container = st.container()
                with pie_container:
                    comp_col = cat_cols[0]
                    comp_data = df.groupby(comp_col)[sales_metric].sum().reset_index()
                    fig1 = px.pie(comp_data, values=sales_metric, names=comp_col,
                                title=f"{sales_metric} by {comp_col}")
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Second composition row - Treemap
                treemap_container = st.container()
                with treemap_container:
                    if len(cat_cols) > 1:
                        fig2 = px.treemap(df, path=cat_cols[:2], values=sales_metric,
                                        title=f"Hierarchical View of {sales_metric}")
                        st.plotly_chart(fig2, use_container_width=True)

    
    # Tab 4: Relationships
    with main_tabs[3]:
        if len(num_cols) >= 2:
            st.subheader("🔗 Correlation Analysis")
            
            # Create a container for the entire relationship analysis
            rel_container = st.container()
            
            with rel_container:
                rel_col1, rel_col2 = st.columns([1, 3])
                
                with rel_col1:
                    # Container for controls and metrics
                    controls_container = st.container()
                    with controls_container:
                        x_col = st.selectbox("X-axis", num_cols)
                        y_col = st.selectbox("Y-axis", [col for col in num_cols if col != x_col])
                        
                        if cat_cols:
                            color_col = st.selectbox("Color by", ['None'] + cat_cols)
                            color_col = None if color_col == 'None' else color_col
                        
                        # Calculate correlation
                        corr = df[[x_col, y_col]].corr().iloc[0,1]
                        
                        # Apply metric card style to correlation coefficient
                        st.markdown("""
                        <div class="metric-container">
                            <p class="metric-title">Correlation Coefficient</p>
                            <p class="metric-value">{:.2f}</p>
                        </div>
                        """.format(corr), unsafe_allow_html=True)
                        
                        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                        direction = "Positive" if corr > 0 else "Negative"
                        if corr != 0:
                            # Apply metric card style to relationship description
                            st.markdown("""
                            <div class="metric-container">
                                <p class="metric-title">Relationship</p>
                                <p class="metric-value">{} {}</p>
                            </div>
                            """.format(strength, direction), unsafe_allow_html=True)
                
                with rel_col2:
                    # Container specifically for the chart
                    chart_container = st.container()
                    with chart_container:
                        if cat_cols and color_col:
                            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                           title=f"{x_col} vs {y_col} by {color_col}")
                        else:
                            fig = px.scatter(df, x=x_col, y=y_col,
                                           title=f"{x_col} vs {y_col}")
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for relationship analysis")
    
    # Tab 5: Advanced Analytics
    with main_tabs[4]:
        st.subheader("🔍 Advanced Analytics")
        
        adv_tab_names = []
        if len(num_cols) >= 2: adv_tab_names.append("Correlation Matrix")
        if date_col and len(num_cols) >= 1: adv_tab_names.append("Time Decomposition") 
        if len(cat_cols) >= 1 and date_col: adv_tab_names.append("Composition Over Time")
        if len(num_cols) >= 3: adv_tab_names.append("3D Visualization")
        if len(cat_cols) >= 2 and sales_metric: adv_tab_names.append("Sunburst Chart")
        
        if adv_tab_names:
            adv_tabs = st.tabs(adv_tab_names)
            tab_idx = 0
            
            # Correlation Matrix
            if "Correlation Matrix" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    # Create a container for correlation matrix
                    corr_container = st.container()
                    
                    with corr_container:
                        selected_num_cols = st.multiselect("Select metrics for correlation", num_cols, num_cols[:5])
                        if len(selected_num_cols) >= 2:
                            corr_matrix = df[selected_num_cols].corr()
                            
                            # Container for the chart
                            chart_container = st.container()
                            with chart_container:
                                fig = px.imshow(corr_matrix,
                                              text_auto=True,
                                              aspect="auto",
                                              color_continuous_scale='RdBu',
                                              title="Correlation Matrix")
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Add styled metric for strongest correlation
                            if len(selected_num_cols) >= 2:
                                corr_values = corr_matrix.values
                                np.fill_diagonal(corr_values, 0)  # Ignore self-correlations
                                max_corr_idx = np.unravel_index(np.abs(corr_values).argmax(), corr_values.shape)
                                max_corr = corr_values[max_corr_idx]
                                col1, col2 = selected_num_cols[max_corr_idx[0]], selected_num_cols[max_corr_idx[1]]
                                
                                st.markdown("""
                                <div class="metric-container">
                                    <p class="metric-title">Strongest Correlation</p>
                                    <p class="metric-value">{:.2f}</p>
                                    <p>between <b>{}</b> and <b>{}</b></p>
                                </div>
                                """.format(max_corr, col1, col2), unsafe_allow_html=True)
                tab_idx += 1
            
            # Time Decomposition
            if "Time Decomposition" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    # Create a container for time decomposition
                    decomp_container = st.container()
                    
                    with decomp_container:
                        ts_col = st.selectbox("Select metric to decompose", num_cols, key="ts_col_select") 
                        try:
                            ts_df = df.set_index(date_col)[ts_col].resample('D').sum().ffill()
                            decomposition = seasonal_decompose(ts_df, model='additive', period=7)
                            
                            # Container for the chart
                            chart_container = st.container()
                            with chart_container:
                                fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
                                fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df, name='Observed'), 
                                            row=1, col=1)
                                fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'), 
                                            row=2, col=1)
                                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'), 
                                            row=3, col=1)
                                fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'), 
                                            row=4, col=1)
                                fig.update_layout(height=600, title_text="Time Series Decomposition")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Add styled metrics for decomposition components
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                trend_strength = 1 - (decomposition.resid.var() / (decomposition.trend.var() + decomposition.resid.var()))
                                st.markdown("""
                                <div class="metric-container">
                                    <p class="metric-title">Trend Strength</p>
                                    <p class="metric-value">{:.2f}</p>
                                </div>
                                """.format(trend_strength), unsafe_allow_html=True)
                                
                            with col2:
                                seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal.var() + decomposition.resid.var()))
                                st.markdown("""
                                <div class="metric-container">
                                    <p class="metric-title">Seasonal Strength</p>
                                    <p class="metric-value">{:.2f}</p>
                                </div>
                                """.format(seasonal_strength), unsafe_allow_html=True)
                                
                            with col3:
                                residual_strength = decomposition.resid.std() / ts_df.std()
                                st.markdown("""
                                <div class="metric-container">
                                    <p class="metric-title">Residual Ratio</p>
                                    <p class="metric-value">{:.2f}</p>
                                </div>
                                """.format(residual_strength), unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.warning(f"Couldn't decompose: {str(e)}")
                tab_idx += 1
            
            # Composition Over Time
            if "Composition Over Time" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    # Create a container for composition over time
                    comp_time_container = st.container()
                    
                    with comp_time_container:
                        comp_col = st.selectbox("Select category", cat_cols, key="time_comp_col_select")
                        metric_col = st.selectbox("Select metric", num_cols, key="time_comp_metric_select")
                        
                        # Container for the chart
                        chart_container = st.container()
                        with chart_container:
                            comp_df = df.groupby([date_col, comp_col])[metric_col].sum().unstack().fillna(0)
                            fig = px.area(comp_df, title=f"{metric_col} Composition by {comp_col} Over Time")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Add metric cards for composition analysis
                        if not comp_df.empty:
                            # Calculate dominant category for each time period
                            dominant_categories = comp_df.idxmax(axis=1)
                            most_frequent = dominant_categories.value_counts().idxmax()
                            
                            # Calculate category with highest growth
                            if len(comp_df) > 1:
                                first_period = comp_df.iloc[0]
                                last_period = comp_df.iloc[-1]
                                growth_pct = (last_period - first_period) / first_period.replace(0, 1) * 100
                                fastest_growing = growth_pct.idxmax()
                                growth_val = growth_pct.max()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("""
                                    <div class="metric-container">
                                        <p class="metric-title">Dominant Category</p>
                                        <p class="metric-value">{}</p>
                                        <p>Most frequently leading</p>
                                    </div>
                                    """.format(most_frequent), unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("""
                                    <div class="metric-container">
                                        <p class="metric-title">Fastest Growing</p>
                                        <p class="metric-value">{}</p>
                                        <p>{:.1f}% growth</p>
                                    </div>
                                    """.format(fastest_growing, growth_val), unsafe_allow_html=True)
                tab_idx += 1
            
            # 3D Visualization
            if "3D Visualization" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    # Create a container for 3D visualization
                    viz3d_container = st.container()
                    
                    with viz3d_container:
                        col3d1, col3d2 = st.columns([1, 3])
                        
                        with col3d1:
                            # Container for controls
                            controls_container = st.container()
                            with controls_container:
                                x_col = st.selectbox("X axis", num_cols, key="3d_x_col")
                                y_col = st.selectbox("Y axis", [c for c in num_cols if c != x_col], key="3d_y_col")
                                z_col = st.selectbox("Z axis", [c for c in num_cols if c not in [x_col, y_col]], key="3d_z_col")
                                
                                if cat_cols:
                                    color_col = st.selectbox("Color by", ['None'] + cat_cols, key="3d_color_col")
                                    color_col = None if color_col == 'None' else color_col
                                
                                # Add styled metric cards for 3D visualization
                                st.markdown("""
                                <div class="metric-container">
                                    <p class="metric-title">Data Points</p>
                                    <p class="metric-value">{:,}</p>
                                </div>
                                """.format(len(df)), unsafe_allow_html=True)
                                
                                # Calculate and display correlation between the three dimensions
                                corr_xy = df[[x_col, y_col]].corr().iloc[0,1]
                                corr_xz = df[[x_col, z_col]].corr().iloc[0,1]
                                corr_yz = df[[y_col, z_col]].corr().iloc[0,1]
                                
                                st.markdown("""
                                <div class="metric-container">
                                    <p class="metric-title">Correlations</p>
                                    <p><b>{} & {}:</b> {:.2f}</p>
                                    <p><b>{} & {}:</b> {:.2f}</p>
                                    <p><b>{} & {}:</b> {:.2f}</p>
                                </div>
                                """.format(
                                    x_col, y_col, corr_xy,
                                    x_col, z_col, corr_xz,
                                    y_col, z_col, corr_yz
                                ), unsafe_allow_html=True)
                        
                        with col3d2:
                            # Container for the chart
                            chart_container = st.container()
                            with chart_container:
                                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                                                 title=f"3D Visualization")
                                st.plotly_chart(fig, use_container_width=True)
                tab_idx += 1
            
            # Sunburst Chart
            if "Sunburst Chart" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    # Create a container for sunburst chart
                    sunburst_container = st.container()
                    
                    with sunburst_container:
                        path_cols = st.multiselect("Select hierarchy path", cat_cols, cat_cols[:2])
                        if path_cols:
                            # Container for the chart
                            chart_container = st.container()
                            with chart_container:
                                fig = px.sunburst(df, path=path_cols, values=sales_metric,
                                                 title=f"Hierarchical View")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Add styled metrics for hierarchy analysis
                            if len(path_cols) >= 2:
                                # Calculate top path
                                grouped = df.groupby(path_cols)[sales_metric].sum().reset_index()

def main():
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

    # Continue with EDA
    # Continue with EDA
    # REMOVE this line: df = load_data(df)
    col_types = detect_column_types(df)
    df = create_filters(df, col_types)

    st.sidebar.success(f"✅ {len(df)} rows × {len(df.columns)} columns")
    st.header(f"Dataset Overview: {dataset_name}")

    show_kpi_cards(generate_kpis(df, col_types))
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Preview", "✨ Smart Visuals", "📊 Correlations", 
        "🧩 Missing Values", 
    ])

    with tab1:
        show_data_preview(df)
    with tab2:
        show_correlation_visualizations(df, col_types)    
    with tab3:
        show_correlations(df, col_types)
    with tab4:
        show_missing_values(df)
