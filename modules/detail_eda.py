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


def apply_dark_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #1E2130;
    }
    h1, h2, h3, h4, h5, h6, .st-b7 {
        color: #FAFAFA !important;
    }
    .st-bb, .st-c0 {
        background-color: #1E2130;
    }
    .stDataFrame {
        background-color: #1E2130;
    }
    .css-1aumxhk {
        background-color: #1E2130;
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    """Load data from uploaded file or default CSV with caching"""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv("Coronation Bakery Dataset.csv")  # Default dataset
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
    
def convert_to_datetime(df):
    """Convert object columns to datetime with explicit format"""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d', 
                          '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M %p']:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt)
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            st.sidebar.success(f"Converted {col} to datetime using format: {fmt}")
                            break
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to datetime: {str(e)}")
    return df

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
    """Apply custom CSS to style metric cards with transparency."""
    st.markdown(
        """
        <style>
            /* Transparent KPI Card Background */
            div[data-testid="metric-container"] {
                background: rgba(255, 255, 255, 0.1); /* Light transparency */
                border-radius: 10px;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            }
            /* Center align the text inside KPI cards */
            div[data-testid="metric-container"] > div {
                align-items: center;
                justify-content: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def show_kpi_cards(kpis):
    """Display dynamic KPI cards"""
    cols = st.columns(len(kpis))
    for i, kpi in enumerate(kpis):
        cols[i].metric(
            label=kpi['title'],
            value=kpi['value'],
            delta=kpi['delta']
        )
    style_metric_cards()

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
            color = 'red' if abs(val) > 0.7 else 'orange' if abs(val) > 0.5 else 'black'
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
        
        - **+1 to +0.7**: Very strong positive relationship
        - **+0.7 to +0.3**: Positive relationship
        - **-0.3 to +0.3**: Little to no relationship
        - **-0.3 to -0.7**: Negative relationship
        - **-0.7 to -1**: Very strong negative relationship
        
        **Method Notes**:
        - **Pearson**: Measures linear relationships (for continuous variables)
        - **Spearman**: Measures monotonic relationships (for ordinal/ranked data)
        - **Kendall**: Similar to Spearman but more robust for small samples
        
        **Note for categorical variables**:
        - Correlations with label-encoded categorical variables should be interpreted with caution
        - The strength depends on how categories are distributed
        - Consider using other statistical tests for categorical-categorical relationships
        """)
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
            df = df[(df[selected_date_col] >= start_date) & 
                    (df[selected_date_col] <= end_date)]
            
    for col in col_types['numeric'][:3]:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        values = st.sidebar.slider(
            f"Filter {col}",
            min_val, max_val, (min_val, max_val)
        )
        df = df[(df[col] >= values[0]) & (df[col] <= values[1])]
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
    
    # Top-level metrics
    if sales_metric:
        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sales", f"{df[sales_metric].sum():,.2f}")
            with col2:
                st.metric("Average Sale", f"{df[sales_metric].mean():,.2f}")
            with col3:
                st.metric("Transactions", f"{len(df):,}")
    
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
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            top_items = df.groupby(col)[sales_metric].sum().nlargest(5).reset_index()
                            fig = px.bar(top_items, x=col, y=sales_metric, 
                                        color=col, title=f"Top {col} by {sales_metric}")
                            st.plotly_chart(fig, use_container_width=True, key=f"top_performers_chart_{i}")
                        with col2:
                            st.markdown("### Key Insight")
                            st.write(f"**{top_items.iloc[0][col]}** leads with **{top_items.iloc[0][sales_metric]:,.2f}** in {sales_metric}.")
                            
                            # Show proportion of top performer
                            total = df[sales_metric].sum()
                            proportion = top_items.iloc[0][sales_metric] / total
                            st.progress(proportion)
                            st.write(f"Represents {proportion:.1%} of total")
        else:
            st.warning("No sales metric columns found for performance analysis")
                
    # Tab 2: Trend Analysis
    with main_tabs[1]:
        date_col = get_date_column(df)
        if date_col and sales_metric:
            st.subheader("📈 Temporal Trends")
            
            trend_col1, trend_col2 = st.columns([1, 3])
            
            with trend_col1:
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
                st.write(f"Peak: **{peak_time}** with **{peak_val:,.2f}**")
                
                # Calculate growth
                first_val = trend_data.iloc[0][selected_metric]
                last_val = trend_data.iloc[-1][selected_metric]
                growth = (last_val - first_val) / first_val if first_val != 0 else 0
                st.metric("Period Growth", f"{growth:.1%}", f"{growth*100:.1f}%")
                
            with trend_col2:
                fig = px.line(trend_data, x=x_col, y=selected_metric, 
                             title=f"{time_groups} Trend of {selected_metric}")
                st.plotly_chart(fig, use_container_width=True, key="trend_line_chart")
        else:
            st.warning("Date column or sales metric not found for trend analysis")
    
    # Tab 3: Distributions
    with main_tabs[2]:
        st.subheader("📊 Distribution Analysis")
        
        num_cols = col_types['numeric'] if 'numeric' in col_types else df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = col_types['categorical'] if 'categorical' in col_types else df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if num_cols:
            dist_col1, dist_col2 = st.columns([1, 3])
            
            with dist_col1:
                dist_col = st.selectbox("Select metric", num_cols)
                if cat_cols:
                    group_col = st.selectbox("Group by", ['None'] + cat_cols)
                
                st.markdown("### Statistics")
                stats = df[dist_col].describe()
                st.dataframe(pd.DataFrame(stats).T.style.highlight_max(axis=1, color='#fffd75'))
            
            with dist_col2:
                if cat_cols and group_col != 'None':
                    fig = px.box(df, x=group_col, y=dist_col, color=group_col,
                                title=f"Distribution of {dist_col} by {group_col}")
                else:
                    fig = px.histogram(df, x=dist_col, 
                                      title=f"Distribution of {dist_col}")
                st.plotly_chart(fig, use_container_width=True, key="distribution_chart")
        else:
            st.warning("No numeric columns found for distribution analysis")
            
        if cat_cols and sales_metric:
            st.subheader("🍰 Composition Analysis")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                comp_col = st.selectbox("Select category", cat_cols, key="comp_col_select")
                comp_data = df.groupby(comp_col)[sales_metric].sum().reset_index()
                fig1 = px.pie(comp_data, values=sales_metric, names=comp_col,
                             title=f"{sales_metric} by {comp_col}")
                st.plotly_chart(fig1, use_container_width=True, key="composition_pie_chart")
            
            with comp_col2:
                if len(cat_cols) > 1:
                    fig2 = px.treemap(df, path=cat_cols[:2], values=sales_metric,
                                     title=f"Hierarchical View of {sales_metric}")
                    st.plotly_chart(fig2, use_container_width=True, key="composition_treemap_chart")
    
    # Tab 4: Relationships
    with main_tabs[3]:
        if len(num_cols) >= 2:
            st.subheader("🔗 Correlation Analysis")
            
            rel_col1, rel_col2 = st.columns([1, 3])
            
            with rel_col1:
                x_col = st.selectbox("X-axis", num_cols)
                y_col = st.selectbox("Y-axis", [col for col in num_cols if col != x_col])
                
                if cat_cols:
                    color_col = st.selectbox("Color by", ['None'] + cat_cols)
                    color_col = None if color_col == 'None' else color_col
                
                # Calculate correlation
                corr = df[[x_col, y_col]].corr().iloc[0,1]
                st.markdown("### Correlation")
                st.metric("Correlation Coefficient", f"{corr:.2f}")
                
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                direction = "Positive" if corr > 0 else "Negative"
                if corr != 0:
                    st.write(f"{strength} {direction} relationship")
            
            with rel_col2:
                if cat_cols and color_col:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                   title=f"{x_col} vs {y_col} by {color_col}")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col,
                                   title=f"{x_col} vs {y_col}")
                    
                st.plotly_chart(fig, use_container_width=True, key="correlation_scatter_chart")
        else:
            st.warning("Need at least 2 numeric columns for relationship analysis")
    
    # Tab 5: Advanced Analytics
    with main_tabs[4]:
        st.subheader("🔍 Advanced Analytics")
        
        adv_tab_names = []
        if len(num_cols) >= 2: adv_tab_names.append("Correlation Matrix")
        if len(num_cols) >= 1: adv_tab_names.append("Normalized Metrics")
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
                    selected_num_cols = st.multiselect("Select metrics for correlation", num_cols, num_cols[:5])
                    if len(selected_num_cols) >= 2:
                        corr_matrix = df[selected_num_cols].corr()
                        fig = px.imshow(corr_matrix,
                                      text_auto=True,
                                      aspect="auto",
                                      color_continuous_scale='RdBu',
                                      title="Correlation Matrix")
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True, key="correlation_matrix_chart")
                tab_idx += 1
            
            # Normalized Metrics
            if "Normalized Metrics" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    selected_metrics = st.multiselect("Select metrics to compare", num_cols, num_cols[:3])
                    if len(selected_metrics) >= 1:
                        norm_df = df[selected_metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

                        try:
                            if date_col and date_col in df.columns:
                                norm_df[date_col] = df[date_col]
                                # Optional: convert to datetime if it looks like a date
                                if "date" in date_col.lower() or "time" in date_col.lower():
                                    norm_df[date_col] = pd.to_datetime(norm_df[date_col], errors="coerce")

                                melt_df = norm_df.melt(id_vars=date_col, var_name='Metric', value_name='Value')
                                fig = px.line(
                                    melt_df,
                                    x=date_col,
                                    y='Value',
                                    color='Metric',
                                    title="Normalized Metric Comparison",
                                    line_shape="spline"
                                )
                            else:
                                melt_df = norm_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
                                fig = px.line(
                                    melt_df,
                                    x='index',
                                    y='Value',
                                    color='Metric',
                                    title="Normalized Metric Comparison"
                                )

                            st.plotly_chart(fig, use_container_width=True, key="normalized_metrics_chart")

                        except Exception as e:
                            st.error(f"Error generating normalized metric comparison plot: {e}")
                            st.dataframe(norm_df.head())  # Help with debugging

                tab_idx += 1

            # Time Decomposition
            if "Time Decomposition" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    ts_col = st.selectbox("Select metric to decompose", num_cols, key="ts_col_select") 
                    try:
                        ts_df = df.set_index(date_col)[ts_col].resample('D').sum().ffill()
                        decomposition = seasonal_decompose(ts_df, model='additive', period=7)
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
                        st.plotly_chart(fig, use_container_width=True, key="time_decomposition_chart")
                    except Exception as e:
                        st.warning(f"Couldn't decompose: {str(e)}")
                tab_idx += 1
            
            # Composition Over Time
            if "Composition Over Time" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    comp_col = st.selectbox("Select category", cat_cols, key="time_comp_col_select")
                    metric_col = st.selectbox("Select metric", num_cols, key="time_comp_metric_select")
                    comp_df = df.groupby([date_col, comp_col])[metric_col].sum().unstack().fillna(0)
                    fig = px.area(comp_df, title=f"{metric_col} Composition by {comp_col} Over Time")
                    st.plotly_chart(fig, use_container_width=True, key="composition_time_chart")
                tab_idx += 1
            
            # 3D Visualization
            if "3D Visualization" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    col3d1, col3d2 = st.columns([1, 3])
                    with col3d1:
                        x_col = st.selectbox("X axis", num_cols, key="3d_x_col")
                        y_col = st.selectbox("Y axis", [c for c in num_cols if c != x_col], key="3d_y_col")
                        z_col = st.selectbox("Z axis", [c for c in num_cols if c not in [x_col, y_col]], key="3d_z_col")
                        
                        if cat_cols:
                            color_col = st.selectbox("Color by", ['None'] + cat_cols, key="3d_color_col")
                            color_col = None if color_col == 'None' else color_col
                    
                    with col3d2:
                        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                                         title=f"3D Visualization")
                        st.plotly_chart(fig, use_container_width=True, key="3d_visualization_chart")
                tab_idx += 1
            
            # Sunburst Chart
            if "Sunburst Chart" in adv_tab_names:
                with adv_tabs[tab_idx]:
                    path_cols = st.multiselect("Select hierarchy path", cat_cols, cat_cols[:2])
                    if path_cols:
                        fig = px.sunburst(df, path=path_cols, values=sales_metric,
                                         title=f"Hierarchical View")
                        st.plotly_chart(fig, use_container_width=True, key="sunburst_chart")

def main():
    apply_dark_theme()
    
    # Get current query parameters
    query_params = st.experimental_get_query_params()
    
    # Set default tab
    tabs = ["📋 Preview", "🔍 Distributions", "📈 Time Series", 
           "🧩 Missing Values", "📊 Correlations", "✨ Smart Visuals"]
    default_tab = tabs[0]
    
    # Get current tab from query params
    current_tab = query_params.get("tab", [default_tab])[0]
    if current_tab not in tabs:
        current_tab = default_tab
    
    st.title("📊 Detailed Exploratory Data Analysis")
    
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
    df = convert_to_datetime(df)
    col_types = detect_column_types(df)
    df = create_filters(df, col_types)

    st.sidebar.success(f"✅ {len(df)} rows × {len(df.columns)} columns")
    st.header(f"Dataset Overview: {dataset_name}")

    show_kpi_cards(generate_kpis(df, col_types))
    
    # Create tabs navigation
    selected_tab = st.radio(
        "Navigation",
        tabs,
        index=tabs.index(current_tab),
        horizontal=True,
        label_visibility="hidden"
    )
    
    # Update query params if tab changed
    if selected_tab != current_tab:
        st.experimental_set_query_params(tab=selected_tab)
        st.experimental_rerun()
    
    # Show the selected tab content
    if selected_tab == "📋 Preview":
        show_data_preview(df)
    elif selected_tab == "🔍 Distributions":
        show_distributions(df, col_types)
    elif selected_tab == "📈 Time Series":
        show_time_series(df, col_types)
    elif selected_tab == "🧩 Missing Values":
        show_missing_values(df)
    elif selected_tab == "📊 Correlations":
        show_correlations(df, col_types)
    elif selected_tab == "✨ Smart Visuals":
        show_correlation_visualizations(df, col_types)
