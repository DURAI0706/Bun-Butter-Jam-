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

# Theme detection and configuration
def get_theme_config():
    """Detect Streamlit theme and return appropriate color settings"""
    # Try to get the current theme from Streamlit config
    try:
        theme = st._config.get_option("theme.base")
        if theme == "dark":
            return {
                'bg_color': 'rgba(0, 0, 0, 0)',
                'text_color': '#FFFFFF',
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'grid_color': 'rgba(255, 255, 255, 0.1)',
                'font_color': 'white',
                'metric_label_color': 'white',
                'metric_value_color': 'white',
                'chart_colors': px.colors.sequential.Plasma,
                'heatmap_colorscale': 'Plasma',
                'table_header_color': 'rgba(49, 51, 63, 0.6)',
                'table_cell_color': 'rgba(49, 51, 63, 0.2)',
                'primary_color': '#3498db',
                'secondary_color': '#2ecc71',
                'accent_color': '#f39c12',
                'divider_color': 'rgba(255, 255, 255, 0.1)'
            }
    except:
        pass
    
    # Default to light theme
    return {
        'bg_color': 'rgba(255, 255, 255, 0)',
        'text_color': '#31333F',
        'plot_bgcolor': 'rgba(255, 255, 255, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0)',
        'grid_color': 'rgba(0, 0, 0, 0.1)',
        'font_color': 'black',
        'metric_label_color': '#31333F',
        'metric_value_color': '#31333F',
        'chart_colors': px.colors.sequential.Blues,
        'heatmap_colorscale': 'Blues',
        'table_header_color': 'rgba(221, 221, 221, 0.6)',
        'table_cell_color': 'rgba(221, 221, 221, 0.2)',
        'primary_color': '#3498db',
        'secondary_color': '#2ecc71',
        'accent_color': '#f39c12',
        'divider_color': 'rgba(0, 0, 0, 0.1)'
    }

# Apply theme-aware styling
def apply_professional_theme(theme_config):
    """Apply a professional theme with custom CSS that adapts to light/dark mode"""
    is_dark = theme_config['text_color'] == '#FFFFFF'
    
    background_color = 'rgb(14, 17, 23)' if is_dark else '#f8f9fa'
    sidebar_background = 'rgb(32, 34, 37)' if is_dark else '#2c3e50'
    text_color = theme_config['text_color']
    primary_color = theme_config['primary_color']
    secondary_color = theme_config['secondary_color']
    accent_color = theme_config['accent_color']
    
    st.markdown(
        f"""
        <style>
            /* Main page styling */
            .main {{
                background-color: {background_color};
                color: {text_color};
            }}
            
            /* Sidebar styling */
            .sidebar .sidebar-content {{
                background-color: {sidebar_background};
                color: white;
            }}
            
            /* Title styling */
            h1, h2, h3, h4, h5, h6 {{
                color: {text_color};
            }}
            
            /* Text styling */
            p, div, span, label {{
                color: {text_color} !important;
            }}
            
            /* Tab styling */
            .stTabs [role="tablist"] {{
                background-color: {'#1e1e1e' if is_dark else '#f1f3f6'};
                padding: 8px 0;
                border-radius: 8px;
            }}
            
            .stTabs [role="tab"][aria-selected="true"] {{
                background-color: {primary_color};
                color: white;
                border-radius: 8px;
                font-weight: bold;
            }}
            
            /* Button styling */
            .stButton button {{
                background-color: {primary_color};
                color: white;
                border-radius: 4px;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
            }}
            
            .stButton button:hover {{
                background-color: {'#2980b9' if not is_dark else '#1a6ea0'};
                color: white;
            }}
            
            /* Dataframe styling */
            .stDataFrame {{
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                background-color: {'rgb(25, 28, 36)' if is_dark else 'white'};
            }}
            
            /* Table styling */
            table {{
                color: {text_color} !important;
            }}
            
            /* Metric card styling */
            div[data-testid="metric-container"] {{
                background: {'rgb(25, 28, 36)' if is_dark else 'white'};
                border-radius: 8px;
                padding: 15px;
                border: 1px solid {'rgb(45, 48, 56)' if is_dark else '#e1e4e8'};
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            
            /* Selectbox styling */
            .stSelectbox, .stMultiselect {{
                background-color: {'rgb(25, 28, 36)' if is_dark else 'white'};
                border-radius: 4px;
                color: {text_color};
            }}
            
            /* Slider styling */
            .stSlider {{
                color: {primary_color};
            }}
            
            /* Info/warning boxes */
            .stAlert {{
                border-radius: 8px;
                background-color: {'rgb(25, 28, 36)' if is_dark else 'white'};
            }}
            
            /* Plotly chart containers */
            .plotly-graph-div {{
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                background: {'rgb(25, 28, 36)' if is_dark else 'white'};
            }}
            
            /* Input fields */
            .stTextInput input, .stNumberInput input, .stTextArea textarea {{
                background-color: {'rgb(25, 28, 36)' if is_dark else 'white'} !important;
                color: {text_color} !important;
            }}
            
            /* Divider color */
            hr {{
                border-color: {theme_config['divider_color']} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def style_metric_cards(theme_config):
    """Apply custom CSS to style metric cards with theme awareness."""
    is_dark = theme_config['text_color'] == '#FFFFFF'
    
    st.markdown(
        f"""
        <style>
            /* Professional KPI Card Styling */
            div[data-testid="metric-container"] {{
                background: {'rgb(25, 28, 36)' if is_dark else 'white'};
                border-radius: 8px;
                padding: 15px;
                border: 1px solid {'rgb(45, 48, 56)' if is_dark else '#e1e4e8'};
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }}
            
            div[data-testid="metric-container"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            
            /* Center align the text inside KPI cards */
            div[data-testid="metric-container"] > div {{
                align-items: center;
                justify-content: center;
            }}
            
            /* Metric title styling */
            div[data-testid="metric-container"] label {{
                font-size: 14px;
                font-weight: 600;
                color: {'#95a5a6' if is_dark else '#7f8c8d'};
            }}
            
            /* Metric value styling */
            div[data-testid="metric-container"] div {{
                font-size: 24px;
                font-weight: 700;
                color: {theme_config['text_color']};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

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
            'delta': None,
            'help': f"Sum of all values in {col} column"
        })
        kpis.append({
            'title': f"Avg {col}",
            'value': f"{df[col].mean():,.2f}",
            'delta': None,
            'help': f"Average value in {col} column"
        })
    for col in col_types['categorical'][:2]:
        kpis.append({
            'title': f"Unique {col}",
            'value': f"{df[col].nunique():,}",
            'delta': None,
            'help': f"Number of unique values in {col} column"
        })
    if col_types['datetime']:
        date_col = col_types['datetime'][0]
        date_range = df[date_col].max() - df[date_col].min()
        kpis.append({
            'title': "Date Range",
            'value': f"{date_range.days} days",
            'delta': None,
            'help': f"Time period covered from {df[date_col].min().date()} to {df[date_col].max().date()}"
        })
    return kpis

def show_kpi_cards(kpis, theme_config):
    """Display dynamic KPI cards with improved layout"""
    st.subheader("📊 Key Metrics")
    cols = st.columns(len(kpis))
    for i, kpi in enumerate(kpis):
        with cols[i]:
            st.metric(
                label=kpi['title'],
                value=kpi['value'],
                delta=kpi['delta'],
                help=kpi.get('help', '')
            )
    style_metric_cards(theme_config)

def show_missing_values(df, theme_config):
    """Show missing values analysis with enhanced visuals"""
    st.subheader("🔍 Missing Values Analysis")
    
    missing = df.isna().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        st.success("✅ No missing values found in the dataset!")
        return
    
    with st.expander("Missing Values Summary", expanded=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("##### Missing Values by Column")
            missing_df = missing.reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
            missing_df = missing_df.sort_values('Missing %', ascending=False)
            
            # Format the display
            styled_df = missing_df.style.format({
                'Missing Count': '{:,}',
                'Missing %': '{:.1f}%'
            }).background_gradient(subset=['Missing %'], cmap='Oranges')
            
            st.dataframe(styled_df, height=300, use_container_width=True)
        
        with col2:
            st.markdown("##### Missing Values Heatmap")
            fig = px.imshow(df.isna(),
                           color_continuous_scale=theme_config['heatmap_colorscale'],
                           title="",
                           width=800,
                           height=400)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Handling Missing Values", expanded=False):
        st.markdown("""
        **Strategies for handling missing values:**
        
        1. **Deletion**:
           - *Listwise deletion*: Remove entire rows with missing values
           - *Column deletion*: Remove columns with high % of missing values
        
        2. **Imputation**:
           - *Mean/Median*: For numerical columns
           - *Mode*: For categorical columns
           - *Advanced methods*: KNN, regression, or model-based imputation
        
        3. **Flagging**:
           - Create indicator variables for missing values
        """)
        
        if st.button("Show Missing Value Patterns"):
            pattern_fig = px.scatter_matrix(df,
                                          dimensions=missing_df['Column'].tolist()[:4],
                                          color=df.isna().any(axis=1).astype(str),
                                          title="Missing Value Patterns")
            st.plotly_chart(pattern_fig, use_container_width=True)

def show_correlations(df, col_types, theme_config):
    """
    Enhanced correlation analysis with professional presentation
    """
    if len(col_types['numeric']) < 2 and len(col_types['categorical']) < 2:
        st.warning("⚠️ Not enough numeric or categorical columns for correlation analysis (need at least 2 columns)")
        return
    
    st.subheader("📈 Feature Correlation Analysis")
    st.markdown("Explore relationships between variables in your dataset")
    
    df_corr = df.copy()
    
    # Encoding categorical variables
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
    
    # Correlation method selection
    corr_method = st.selectbox(
        "Select correlation method",
        ['pearson', 'spearman', 'kendall'],
        index=0,
        help="Pearson: Linear relationships, Spearman: Monotonic relationships, Kendall: Rank correlations"
    )
    
    # Main correlation display
    corr = df_corr[all_cols].corr(method=corr_method)
    
    tab1, tab2 = st.tabs(["📊 Correlation Matrix", "🔍 Detailed Analysis"])
    
    with tab1:
        st.markdown(f"#### {corr_method.title()} Correlation Matrix")
        
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            aspect="auto",
            labels=dict(color="Correlation"),
            width=800,
            height=600
        )
        
        fig.update_layout(
            margin=dict(l=50, r=50, b=100, t=50),
            xaxis=dict(tickangle=45),
            coloraxis_colorbar=dict(
                title="Correlation",
                thickness=20,
                len=0.75,
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        fig.update_traces(
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Top Correlations")
        
        corr_unstacked = corr.abs().unstack()
        corr_unstacked = corr_unstacked[corr_unstacked < 1]  # Remove self-correlations
        corr_pairs = corr_unstacked.sort_values(ascending=False).reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        
        # Add type information
        dtypes = df.dtypes.astype(str).to_dict()
        corr_pairs['Type 1'] = corr_pairs['Feature 1'].map(dtypes)
        corr_pairs['Type 2'] = corr_pairs['Feature 2'].map(dtypes)
        corr_pairs['Correlation'] = corr_pairs['Correlation'].round(3)
        
        # Remove duplicates
        corr_pairs = corr_pairs.drop_duplicates(subset=['Correlation'])
        
        # Style the dataframe
        def color_high_correlations(val):
            color = 'red' if abs(val) > 0.7 else 'orange' if abs(val) > 0.5 else 'black'
            return f'color: {color}; font-weight: bold;'
        
        display_df = corr_pairs.head(20).copy()
        
        st.dataframe(
            display_df.style.map(color_high_correlations, subset=['Correlation']),
            height=600,
            use_container_width=True
        )
        
        if len(col_types['categorical']) > 0:
            st.info("ℹ️ Note: Categorical columns were label encoded for correlation analysis")
    
    # Interpretation guide
    with st.expander("📖 Correlation Interpretation Guide", expanded=False):
        st.markdown(f"""
        ### How to Interpret {corr_method.title()} Correlations
        
        | Correlation Range | Interpretation                     |
        |-------------------|-----------------------------------|
        | 0.8 to 1.0        | Very strong relationship          |
        | 0.6 to 0.8        | Strong relationship               |
        | 0.4 to 0.6        | Moderate relationship             |
        | 0.2 to 0.4        | Weak relationship                 |
        | 0.0 to 0.2        | Very weak or no relationship      |
        
        **Negative values** indicate an inverse relationship.
        
        **Important Notes**:
        - Correlation does not imply causation
        - Results may be affected by outliers
        - The strength of correlation needed to be 'significant' depends on your field/context
        - For categorical variables, consider using other statistical tests like Chi-square
        """)

def show_distributions(df, col_types, theme_config):
    """Enhanced distribution analysis with professional presentation"""
    st.subheader("📊 Data Distributions")
    st.markdown("Explore the distribution of values in your dataset")
    
    if col_types['numeric']:
        st.markdown("### Numeric Variables")
        
        num_col = st.selectbox(
            "Select numeric column", 
            col_types['numeric'],
            key='num_dist_select'
        )
        
        tab1, tab2, tab3 = st.tabs(["Histogram", "Box Plot", "Statistics"])
        
        with tab1:
            fig = px.histogram(
                df, 
                x=num_col, 
                title=f"Distribution of {num_col}",
                marginal="box",
                nbins=50,
                color_discrete_sequence=[theme_config['primary_color']]
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=num_col,
                yaxis_title="Count",
                plot_bgcolor=theme_config['plot_bgcolor'],
                paper_bgcolor=theme_config['paper_bgcolor'],
                font=dict(color=theme_config['font_color'])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.box(
                df, 
                y=num_col, 
                title=f"Box Plot of {num_col}",
                color_discrete_sequence=[theme_config['primary_color']]
            )
            fig.update_layout(
                plot_bgcolor=theme_config['plot_bgcolor'],
                paper_bgcolor=theme_config['paper_bgcolor'],
                font=dict(color=theme_config['font_color'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate outliers
            q1 = df[num_col].quantile(0.25)
            q3 = df[num_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[num_col] < lower_bound) | (df[num_col] > upper_bound)]
            
            st.info(f"**Outlier Analysis**: Detected {len(outliers):,} outliers ({len(outliers)/len(df):.1%} of data) using the IQR method")
        
        with tab3:
            stats = df[num_col].describe().to_frame().T
            stats['skewness'] = df[num_col].skew()
            stats['kurtosis'] = df[num_col].kurtosis()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Descriptive Statistics")
                st.dataframe(
                    stats.style.format("{:.2f}"),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("##### Distribution Shape")
                
                skew_val = stats['skewness'].iloc[0]
                if abs(skew_val) < 0.5:
                    skew_desc = "Approximately symmetric"
                elif 0.5 <= skew_val < 1 or -1 < skew_val <= -0.5:
                    skew_desc = "Moderately skewed"
                else:
                    skew_desc = "Highly skewed"
                
                kurt_val = stats['kurtosis'].iloc[0]
                if kurt_val > 3:
                    kurt_desc = "Leptokurtic (heavy-tailed)"
                elif kurt_val < 3:
                    kurt_desc = "Platykurtic (light-tailed)"
                else:
                    kurt_desc = "Mesokurtic (normal-like)"
                
                st.markdown(f"""
                - **Skewness**: {skew_val:.2f} ({skew_desc})
                - **Kurtosis**: {kurt_val:.2f} ({kurt_desc})
                """)
    
    if col_types['categorical']:
        st.markdown("### Categorical Variables")
        
        cat_col = st.selectbox(
            "Select categorical column", 
            col_types['categorical'],
            key='cat_dist_select'
        )
        
        tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
        
        with tab1:
            value_counts = df[cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']
            
            fig = px.bar(
                value_counts, 
                x='Value', 
                y='Count', 
                title=f"Distribution of {cat_col}",
                color='Value',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                plot_bgcolor=theme_config['plot_bgcolor'],
                paper_bgcolor=theme_config['paper_bgcolor'],
                font=dict(color=theme_config['font_color'])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.pie(
                value_counts, 
                values='Count', 
                names='Value',
                title=f"Proportion of {cat_col}",
                hole=0.3
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#ffffff', width=1))
            )
            fig.update_layout(
                plot_bgcolor=theme_config['plot_bgcolor'],
                paper_bgcolor=theme_config['paper_bgcolor'],
                font=dict(color=theme_config['font_color'])
            )
            st.plotly_chart(fig, use_container_width=True)

def show_time_series(df, col_types, theme_config):
    """Enhanced time series analysis with professional presentation"""
    if not col_types['datetime']:
        st.warning("No datetime columns found for time series analysis")
        return
    
    st.subheader("⏳ Time Series Analysis")
    st.markdown("Analyze trends and patterns over time")
    
    # Column selection
    col1, col2 = st.columns(2)
    
    with col1:
        date_col = st.selectbox(
            "Select date column", 
            col_types['datetime'], 
            key='ts_date_col'
        )
    
    with col2:
        if not col_types['numeric']:
            st.warning("No numeric columns for time series visualization")
            return
        
        value_col = st.selectbox(
            "Select value column", 
            col_types['numeric'], 
            key='ts_value_col'
        )
    
    # Resampling options
    resample_options = {
        'Raw': None,
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'ME',
        'Quarterly': 'QE',
        'Yearly': 'YE'
    }
    
    selected_freq = st.selectbox(
        "Aggregation frequency", 
        list(resample_options.keys()),
        index=2,
        help="Select time aggregation level for analysis"
    )
    
    # Prepare time series data
    ts_df = df.set_index(date_col)[[value_col]]
    
    if resample_options[selected_freq]:
        ts_df = ts_df.resample(resample_options[selected_freq]).mean()
    
    ts_df = ts_df.reset_index()
    
    # Main time series plot
    fig = px.line(
        ts_df, 
        x=date_col, 
        y=value_col, 
        title=f"{value_col} over Time",
        markers=True,
        color_discrete_sequence=[theme_config['primary_color']]
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=value_col,
        hovermode="x unified",
        plot_bgcolor=theme_config['plot_bgcolor'],
        paper_bgcolor=theme_config['paper_bgcolor'],
        font=dict(color=theme_config['font_color'])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Advanced time series analysis
    with st.expander("Advanced Time Series Analysis", expanded=False):
        st.markdown("#### Decomposition Analysis")
        
        if len(ts_df) < 30:
            st.warning("Need at least 30 observations for decomposition")
        else:
            try:
                decomposition = seasonal_decompose(
                    df.set_index(date_col)[value_col].dropna(),
                    model='additive',
                    period=7 if selected_freq == 'Daily' else 12
                )
                
                fig = make_subplots(
                    rows=4, 
                    cols=1, 
                    shared_xaxes=True,
                    subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.observed.index,
                        y=decomposition.observed,
                        name='Observed',
                        line=dict(color=theme_config['primary_color'])
                    ), 
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        name='Trend',
                        line=dict(color=theme_config['secondary_color'])
                    ), 
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal,
                        name='Seasonal',
                        line=dict(color=theme_config['accent_color'])
                    ), 
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.resid.index,
                        y=decomposition.resid,
                        name='Residual',
                        line=dict(color='#f39c12')
                    ), 
                    row=4, col=1
                )
                
                fig.update_layout(
                    height=800,
                    showlegend=False,
                    title_text="Time Series Decomposition",
                    plot_bgcolor=theme_config['plot_bgcolor'],
                    paper_bgcolor=theme_config['paper_bgcolor'],
                    font=dict(color=theme_config['font_color'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in decomposition: {str(e)}")

def create_filters(df, col_types, theme_config):
    """Create professional-looking dynamic filters in sidebar"""
    st.sidebar.header("🔍 Data Filters")
    
    # Date range filter
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if date_cols:
        selected_date_col = st.sidebar.selectbox(
            "Filter by date column", 
            date_cols,
            help="Select a date column to filter by date range"
        )
        
        min_date = df[selected_date_col].min()
        max_date = df[selected_date_col].max()
        
        date_range = st.sidebar.date_input(
            f"Select date range for {selected_date_col}",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter data within this date range"
        )
        
        if len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            df = df[(df[selected_date_col] >= start_date) & 
                    (df[selected_date_col] <= end_date)]
    
    # Numeric filters
    if col_types['numeric']:
        st.sidebar.markdown("### Numeric Filters")
        for col in col_types['numeric'][:3]:  # Limit to 3 numeric filters
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            values = st.sidebar.slider(
                f"Filter {col}",
                min_val, max_val, (min_val, max_val),
                help=f"Filter {col} between {min_val:.2f} and {max_val:.2f}"
            )
            df = df[(df[col] >= values[0]) & (df[col] <= values[1])]
    
    # Categorical filters
    if col_types['categorical']:
        st.sidebar.markdown("### Categorical Filters")
        for col in col_types['categorical'][:3]:  # Limit to 3 categorical filters
            options = df[col].unique().tolist()
            selected = st.sidebar.multiselect(
                f"Filter {col}",
                options,
                default=options,
                help=f"Select values to include for {col}"
            )
            if selected:
                df = df[df[col].isin(selected)]
    
    return df

def show_data_preview(df, theme_config):
    """Enhanced data preview with professional presentation"""
    st.subheader("📋 Data Preview")
    st.markdown("Explore and interact with your dataset")
    
    if len(df) > 10000:
        st.warning("Large dataset detected! Showing sample of 10,000 rows.")
        sample_df = df.sample(10000)
    else:
        sample_df = df.copy()
    
    tab1, tab2, tab3 = st.tabs(["Interactive Explorer", "Descriptive Statistics", "Data Structure"])
    
    with tab1:
        st.markdown("#### Interactive Data Explorer")
        height = min(600, 100 + len(sample_df) * 25)
        filtered_df = dataframe_explorer(sample_df)
        
        st.dataframe(
            filtered_df, 
            use_container_width=True, 
            height=height
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download filtered data",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name='filtered_data.csv',
                mime='text/csv',
                help="Download currently filtered data as CSV"
            )
        
        with col2:
            st.download_button(
                label="📥 Download full dataset",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='full_data.csv',
                mime='text/csv',
                help="Download complete dataset as CSV"
            )
    
    with tab2:
        st.markdown("#### Descriptive Statistics")
        
        stat_tabs = st.tabs(["Numerical", "Categorical", "All Columns"]) 
        
        with stat_tabs[0]:
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                st.dataframe(
                    df[num_cols].describe().T.style.background_gradient(
                        cmap='Blues',
                        subset=['mean', '50%', 'std']
                    ).format("{:.2f}"),
                    use_container_width=True
                )
            else:
                st.warning("No numerical columns found")
        
        with stat_tabs[1]:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                cat_stats = pd.DataFrame({
                    'Unique Values': df[cat_cols].nunique(),
                    'Most Common': df[cat_cols].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'),
                    'Frequency': df[cat_cols].apply(lambda x: x.value_counts().iloc[0] if len(x.value_counts()) > 0 else 0),
                    'Missing Values': df[cat_cols].isna().sum()
                })
                
                st.dataframe(
                    cat_stats.style.background_gradient(
                        cmap='Greens',
                        subset=['Unique Values', 'Frequency']
                    ).format({
                        'Frequency': '{:,}',
                        'Missing Values': '{:,}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("No categorical columns found")
        
        with stat_tabs[2]:
            st.dataframe(
                df.describe(include='all').T.style.background_gradient(
                    cmap='Purples',
                    subset=['mean', '50%', 'freq', 'top']
                ).format({
                    'mean': '{:.2f}',
                    '50%': '{:.2f}',
                    'freq': '{:.0f}'
                }),
                use_container_width=True
            )    
    
    with tab3:
        st.markdown("#### Data Structure")
        
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Missing Values': df.isna().sum(),
            '% Missing': (df.isna().mean() * 100).round(2),
            'Unique Values': df.nunique()
        }).sort_values(by='Type')
        
        st.dataframe(
            dtype_info.style.bar(
                subset=['% Missing'], 
                color='#ff6961'
            ).format({
                '% Missing': '{:.1f}%',
                'Missing Values': '{:,}',
                'Unique Values': '{:,}'
            }),
            use_container_width=True
        )
        
        st.markdown(f"""
        **Dataset Summary:**
        - **Shape**: {df.shape[0]:,} rows × {df.shape[1]:,} columns
        - **Memory Usage**: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB
        """)

def show_correlation_visualizations(df, col_types, theme_config):
    """Enhanced correlation visualizations with professional presentation"""
    st.subheader("✨ Advanced Analytics")
    st.markdown("Interactive visualizations to explore relationships in your data")
    
    # Helper function to identify date columns
    def get_date_column(df):
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            return date_cols[0]        
        for col in df.select_dtypes(include=['datetime']).columns:
            return col
        return None
    
    # Identify key metrics columns
    amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'total' in col.lower()]
    quantity_cols = [col for col in df.columns if 'quantity' in col.lower() or 'qty' in col.lower()]
    sales_metric = amount_cols[0] if amount_cols else (quantity_cols[0] if quantity_cols else None)
    
    # Main dashboard layout
    tab_names = []
    if sales_metric: tab_names.append("Performance")
    if get_date_column(df) and sales_metric: tab_names.append("Trends")
    if len(col_types['numeric']) > 0: tab_names.append("Distributions")
    if len(col_types['numeric']) >= 2: tab_names.append("Relationships")
    if len(col_types['numeric']) >= 1: tab_names.append("Advanced")
    
    if not tab_names:
        st.warning("Not enough columns for advanced visualizations")
        return
    
    main_tabs = st.tabs(tab_names)
    tab_idx = 0
    
    # Performance Tab
    if "Performance" in tab_names:
        with main_tabs[tab_idx]:
            st.markdown("## 🏆 Performance Analysis")
            st.markdown("Analyze top performers and key metrics")
            
            cat_cols = col_types['categorical'] if 'categorical' in col_types else df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                perf_tab_names = [f"By {col}" for col in cat_cols[:3]]  # Limit to 3 categorical dimensions
                perf_tabs = st.tabs(perf_tab_names)
                
                for i, col in enumerate(cat_cols[:3]):
                    with perf_tabs[i]:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Calculate top performers
                            top_items = df.groupby(col)[sales_metric].sum().nlargest(10).reset_index()
                            
                            # Create bar chart
                            fig = px.bar(
                                top_items, 
                                x=col, 
                                y=sales_metric, 
                                color=col,
                                title=f"Top {col} by {sales_metric}",
                                text=sales_metric,
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            
                            # Format the chart
                            fig.update_layout(
                                xaxis_title=col,
                                yaxis_title=sales_metric,
                                showlegend=False,
                                uniformtext_minsize=8,
                                uniformtext_mode='hide',
                                plot_bgcolor=theme_config['plot_bgcolor'],
                                paper_bgcolor=theme_config['paper_bgcolor'],
                                font=dict(color=theme_config['font_color'])
                            )
                            
                            # Format the text on bars
                            fig.update_traces(
                                texttemplate='%{text:,.0f}',
                                textposition='outside'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### Key Insights")
                            
                            # Calculate metrics
                            top_value = top_items.iloc[0][sales_metric]
                            top_item = top_items.iloc[0][col]
                            total = df[sales_metric].sum()
                            proportion = top_value / total
                            
                            # Display metrics
                            st.metric(
                                label=f"Top {col}",
                                value=top_item,
                                delta=f"{top_value:,.0f} ({proportion:.1%} of total)"
                            )
                            
                            # Show distribution
                            st.markdown("##### Distribution")
                            st.vega_lite_chart(top_items, {
                                "mark": {"type": "area", "interpolate": "monotone"},
                                "encoding": {
                                    "x": {"field": col, "type": "ordinal", "sort": f"-{sales_metric}"},
                                    "y": {"field": sales_metric, "type": "quantitative"}
                                }
                            }, use_container_width=True)
            else:
                st.warning("No categorical columns found for performance analysis")
            
            tab_idx += 1
    
    # Trends Tab
    if "Trends" in tab_names:
        with main_tabs[tab_idx]:
            st.markdown("## 📈 Trend Analysis")
            st.markdown("Analyze temporal patterns and seasonality")
            
            date_col = get_date_column(df)
            
            trend_col1, trend_col2 = st.columns([1, 3])
            
            with trend_col1:
                # Controls for trend analysis
                metric_options = []
                if sales_metric: metric_options.append(sales_metric)
                if quantity_cols: metric_options.extend(quantity_cols[:2])
                if amount_cols: metric_options.extend(amount_cols[:2])
                
                selected_metric = st.selectbox(
                    "Select metric", 
                    metric_options, 
                    key="trend_metric_select",
                    help="Select the metric to analyze over time"
                )
                
                time_groups = st.radio(
                    "Time aggregation", 
                    ['Daily', 'Weekly', 'Monthly', 'Quarterly'],
                    key="time_group_select",
                    help="Select the time aggregation level"
                )
                
                st.markdown("### Trend Metrics")
                
                # Calculate trend data based on time grouping
                df[date_col] = pd.to_datetime(df[date_col])
                
                if time_groups == 'Daily':
                    trend_data = df.groupby(df[date_col].dt.date)[selected_metric].sum().reset_index()
                    x_col = date_col
                elif time_groups == 'Weekly':
                    trend_data = df.groupby([
                        df[date_col].dt.year.rename("Year"), 
                        df[date_col].dt.isocalendar().week.rename("Week")
                    ])[selected_metric].sum().reset_index()
                    trend_data['Week'] = trend_data['Year'].astype(str) + '-W' + trend_data['Week'].astype(str).str.zfill(2)
                    x_col = 'Week'
                elif time_groups == 'Monthly':
                    trend_data = df.groupby([
                        df[date_col].dt.year.rename("Year"), 
                        df[date_col].dt.month.rename("Month")
                    ])[selected_metric].sum().reset_index()
                    trend_data['Month'] = trend_data['Year'].astype(str) + '-' + trend_data['Month'].astype(str).str.zfill(2)
                    x_col = 'Month'
                else:  # Quarterly
                    trend_data = df.groupby([
                        df[date_col].dt.year.rename("Year"), 
                        df[date_col].dt.quarter.rename("Quarter")
                    ])[selected_metric].sum().reset_index()
                    trend_data['Quarter'] = trend_data['Year'].astype(str) + '-Q' + trend_data['Quarter'].astype(str)
                    x_col = 'Quarter'
                
                # Calculate metrics
                peak_val = trend_data[selected_metric].max()
                peak_time = trend_data.loc[trend_data[selected_metric].idxmax(), x_col]
                min_val = trend_data[selected_metric].min()
                min_time = trend_data.loc[trend_data[selected_metric].idxmin(), x_col]
                
                st.metric(
                    "Peak Value", 
                    f"{peak_val:,.0f}", 
                    f"Occurred at {peak_time}"
                )
                
                st.metric(
                    "Lowest Value", 
                    f"{min_val:,.0f}", 
                    f"Occurred at {min_time}"
                )
                
                # Calculate growth
                first_val = trend_data.iloc[0][selected_metric]
                last_val = trend_data.iloc[-1][selected_metric]
                growth = (last_val - first_val) / first_val if first_val != 0 else 0
                
                st.metric(
                    "Overall Growth", 
                    f"{growth:.1%}", 
                    delta_color="inverse" if growth < 0 else "normal"
                )
                
            with trend_col2:
                # Create trend chart
                fig = px.line(
                    trend_data, 
                    x=x_col, 
                    y=selected_metric, 
                    title=f"{time_groups} Trend of {selected_metric}",
                    markers=True,
                    color_discrete_sequence=[theme_config['primary_color']]
                )
                
                # Add rolling average
                if len(trend_data) > 10:
                    window = min(6, len(trend_data)//3)
                    trend_data['rolling_avg'] = trend_data[selected_metric].rolling(window=window).mean()
                    
                    fig.add_scatter(
                        x=trend_data[x_col],
                        y=trend_data['rolling_avg'],
                        mode='lines',
                        name=f'{window}-period Moving Avg',
                        line=dict(color='red', dash='dot')
                    )
                
                fig.update_layout(
                    xaxis_title="Time Period",
                    yaxis_title=selected_metric,
                    hovermode="x unified",
                    showlegend=True,
                    plot_bgcolor=theme_config['plot_bgcolor'],
                    paper_bgcolor=theme_config['paper_bgcolor'],
                    font=dict(color=theme_config['font_color'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add seasonal decomposition
                with st.expander("Seasonal Decomposition", expanded=False):
                    if len(trend_data) > 30:
                        try:
                            decomposition = seasonal_decompose(
                                df.set_index(date_col)[selected_metric].dropna(),
                                model='additive',
                                period=7 if time_groups == 'Daily' else 12
                            )
                            
                            decomp_fig = make_subplots(
                                rows=4, 
                                cols=1, 
                                shared_xaxes=True,
                                subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                            )
                            
                            decomp_fig.add_trace(
                                go.Scatter(
                                    x=decomposition.observed.index,
                                    y=decomposition.observed,
                                    name='Observed',
                                    line=dict(color=theme_config['primary_color'])
                                ), 
                                row=1, col=1
                            )
                            
                            decomp_fig.add_trace(
                                go.Scatter(
                                    x=decomposition.trend.index,
                                    y=decomposition.trend,
                                    name='Trend',
                                    line=dict(color=theme_config['secondary_color'])
                                ), 
                                row=2, col=1
                            )
                            
                            decomp_fig.add_trace(
                                go.Scatter(
                                    x=decomposition.seasonal.index,
                                    y=decomposition.seasonal,
                                    name='Seasonal',
                                    line=dict(color=theme_config['accent_color'])
                                ), 
                                row=3, col=1
                            )
                            
                            decomp_fig.add_trace(
                                go.Scatter(
                                    x=decomposition.resid.index,
                                    y=decomposition.resid,
                                    name='Residual',
                                    line=dict(color='#f39c12')
                                ), 
                                row=4, col=1
                            )
                            
                            decomp_fig.update_layout(
                                height=600,
                                showlegend=False,
                                title_text="Time Series Decomposition",
                                plot_bgcolor=theme_config['plot_bgcolor'],
                                paper_bgcolor=theme_config['paper_bgcolor'],
                                font=dict(color=theme_config['font_color'])
                            )
                            
                            st.plotly_chart(decomp_fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error in decomposition: {str(e)}")
                    else:
                        st.warning("Need at least 30 observations for seasonal decomposition")
            
            tab_idx += 1
    
    # Distributions Tab
    if "Distributions" in tab_names:
        with main_tabs[tab_idx]:
            st.markdown("## 📊 Distribution Analysis")
            st.markdown("Explore how values are distributed across your data")
            
            num_cols = col_types['numeric'] if 'numeric' in col_types else df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = col_types['categorical'] if 'categorical' in col_types else df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if num_cols:
                st.markdown("### Numeric Distributions")
                
                dist_col1, dist_col2 = st.columns([1, 3])
                
                with dist_col1:
                    dist_col = st.selectbox(
                        "Select numeric column", 
                        num_cols,
                        key="dist_num_select"
                    )
                    
                    if cat_cols:
                        group_col = st.selectbox(
                            "Group by", 
                            ['None'] + cat_cols,
                            key="dist_group_select"
                        )
                    
                    st.markdown("#### Statistics")
                    stats = df[dist_col].describe().to_frame().T
                    stats['skewness'] = df[dist_col].skew()
                    stats['kurtosis'] = df[dist_col].kurt()
                    
                    st.dataframe(
                        stats.style.format("{:.2f}"),
                        use_container_width=True
                    )
                
                with dist_col2:
                    if cat_cols and group_col != 'None':
                        fig = px.box(
                            df, 
                            x=group_col, 
                            y=dist_col, 
                            color=group_col,
                            title=f"Distribution of {dist_col} by {group_col}",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    else:
                        fig = px.histogram(
                            df, 
                            x=dist_col, 
                            title=f"Distribution of {dist_col}",
                            marginal="box",
                            nbins=50,
                            color_discrete_sequence=[theme_config['primary_color']]
                        )
                    
                    fig.update_layout(
                        showlegend=False,
                        xaxis_title=dist_col,
                        yaxis_title="Count",
                        plot_bgcolor=theme_config['plot_bgcolor'],
                        paper_bgcolor=theme_config['paper_bgcolor'],
                        font=dict(color=theme_config['font_color'])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            if cat_cols and sales_metric:
                st.markdown("### Composition Analysis")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    comp_col = st.selectbox(
                        "Select category", 
                        cat_cols, 
                        key="comp_col_select"
                    )
                    
                    comp_data = df.groupby(comp_col)[sales_metric].sum().reset_index()
                    
                    fig1 = px.pie(
                        comp_data, 
                        values=sales_metric, 
                        names=comp_col,
                        title=f"{sales_metric} by {comp_col}",
                        hole=0.3
                    )
                    
                    fig1.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#ffffff', width=1))
                    )
                    fig1.update_layout(
                        plot_bgcolor=theme_config['plot_bgcolor'],
                        paper_bgcolor=theme_config['paper_bgcolor'],
                        font=dict(color=theme_config['font_color'])
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with comp_col2:
                    if len(cat_cols) > 1:
                        fig2 = px.treemap(
                            df, 
                            path=cat_cols[:2], 
                            values=sales_metric,
                            title=f"Hierarchical View of {sales_metric}"
                        )
                        fig2.update_layout(
                            plot_bgcolor=theme_config['plot_bgcolor'],
                            paper_bgcolor=theme_config['paper_bgcolor'],
                            font=dict(color=theme_config['font_color'])
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
            
            tab_idx += 1
    
    # Relationships Tab
    if "Relationships" in tab_names:
        with main_tabs[tab_idx]:
            st.markdown("## 🔗 Relationship Analysis")
            st.markdown("Explore correlations and relationships between variables")
            
            num_cols = col_types['numeric'] if 'numeric' in col_types else df.select_dtypes(include=np.number).columns.tolist()
            
            if len(num_cols) >= 2:
                rel_col1, rel_col2 = st.columns([1, 3])
                
                with rel_col1:
                    x_col = st.selectbox(
                        "X-axis variable", 
                        num_cols,
                        key="rel_x_select"
                    )
                    
                    y_col = st.selectbox(
                        "Y-axis variable", 
                        [col for col in num_cols if col != x_col],
                        key="rel_y_select"
                    )
                    
                    if cat_cols:
                        color_col = st.selectbox(
                            "Color by", 
                            ['None'] + cat_cols,
                            key="rel_color_select"
                        )
                        color_col = None if color_col == 'None' else color_col
                    
                    # Calculate correlation
                    corr = df[[x_col, y_col]].corr().iloc[0,1]
                    
                    st.markdown("### Correlation")
                    st.metric(
                        "Pearson Correlation", 
                        f"{corr:.2f}",
                        help="Measure of linear relationship (-1 to 1)"
                    )
                    
                    # Interpret correlation
                    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                    direction = "Positive" if corr > 0 else "Negative"
                    
                    st.markdown(f"""
                    **Interpretation**:
                    - **Strength**: {strength}
                    - **Direction**: {direction}
                    """)
                    
                    # Show regression line option
                    show_trendline = st.checkbox(
                        "Show trend line",
                        value=True,
                        key="rel_trend_check"
                    )
                
                with rel_col2:
                    if cat_cols and color_col:
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col, 
                            color=color_col,
                            title=f"{x_col} vs {y_col} by {color_col}",
                            trendline="ols" if show_trendline else None,
                            opacity=0.7,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    else:
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            title=f"{x_col} vs {y_col}",
                            trendline="ols" if show_trendline else None,
                            opacity=0.7,
                            color_discrete_sequence=[theme_config['primary_color']]
                        )
                    
                    fig.update_layout(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        hovermode="closest",
                        plot_bgcolor=theme_config['plot_bgcolor'],
                        paper_bgcolor=theme_config['paper_bgcolor'],
                        font=dict(color=theme_config['font_color'])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Advanced correlation analysis
                with st.expander("Advanced Correlation Analysis", expanded=False):
                    st.markdown("#### Correlation Matrix")
                    
                    selected_num_cols = st.multiselect(
                        "Select variables for matrix", 
                        num_cols, 
                        default=num_cols[:5],
                        key="corr_matrix_select"
                    )
                    
                    if len(selected_num_cols) >= 2:
                        corr_matrix = df[selected_num_cols].corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale='RdBu',
                            zmin=-1,
                            zmax=1,
                            title="Correlation Matrix"
                        )
                        
                        fig.update_layout(
                            height=600,
                            xaxis_title="",
                            yaxis_title="",
                            plot_bgcolor=theme_config['plot_bgcolor'],
                            paper_bgcolor=theme_config['paper_bgcolor'],
                            font=dict(color=theme_config['font_color'])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for relationship analysis")
            
            tab_idx += 1
    
    # Advanced Tab
    if "Advanced" in tab_names:
        with main_tabs[tab_idx]:
            st.markdown("## 🔍 Advanced Analytics")
            st.markdown("Sophisticated techniques for deeper insights")
            
            adv_tab_names = []
            if len(num_cols) >= 2: adv_tab_names.append("Correlation Matrix")
            if len(num_cols) >= 1: adv_tab_names.append("Normalized Metrics")
            if get_date_column(df) and len(num_cols) >= 1: adv_tab_names.append("Time Decomposition") 
            if len(cat_cols) >= 1 and get_date_column(df): adv_tab_names.append("Composition Over Time")
            if len(num_cols) >= 3: adv_tab_names.append("3D Visualization")
            if len(cat_cols) >= 2 and sales_metric: adv_tab_names.append("Sunburst Chart")
            
            if adv_tab_names:
                adv_tabs = st.tabs(adv_tab_names)
                adv_tab_idx = 0
                
                # Correlation Matrix
                if "Correlation Matrix" in adv_tab_names:
                    with adv_tabs[adv_tab_idx]:
                        selected_num_cols = st.multiselect(
                            "Select metrics for correlation", 
                            num_cols, 
                            num_cols[:5],
                            key="adv_corr_matrix_select"
                        )
                        
                        if len(selected_num_cols) >= 2:
                            corr_matrix = df[selected_num_cols].corr()
                            
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=".2f",
                                aspect="auto",
                                color_continuous_scale='RdBu',
                                zmin=-1,
                                zmax=1,
                                title="Correlation Matrix"
                            )
                            
                            fig.update_layout(
                                height=600,
                                plot_bgcolor=theme_config['plot_bgcolor'],
                                paper_bgcolor=theme_config['paper_bgcolor'],
                                font=dict(color=theme_config['font_color'])
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    adv_tab_idx += 1
                
                # Normalized Metrics
                if "Normalized Metrics" in adv_tab_names:
                    with adv_tabs[adv_tab_idx]:
                        selected_metrics = st.multiselect(
                            "Select metrics to compare", 
                            num_cols, 
                            num_cols[:3],
                            key="norm_metrics_select"
                        )
                        
                        if len(selected_metrics) >= 1:
                            norm_df = df[selected_metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
                            
                            try:
                                date_col = get_date_column(df)
                                if date_col and date_col in df.columns:
                                    norm_df[date_col] = df[date_col]
                                    melt_df = norm_df.melt(id_vars=date_col, var_name='Metric', value_name='Value')
                                    
                                    fig = px.line(
                                        melt_df,
                                        x=date_col,
                                        y='Value',
                                        color='Metric',
                                        title="Normalized Metric Comparison",
                                        line_shape="spline",
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                else:
                                    melt_df = norm_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
                                    
                                    fig = px.line(
                                        melt_df,
                                        x='index',
                                        y='Value',
                                        color='Metric',
                                        title="Normalized Metric Comparison",
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                
                                fig.update_layout(
                                    xaxis_title="Time" if date_col else "Index",
                                    yaxis_title="Normalized Value",
                                    legend_title="Metric",
                                    plot_bgcolor=theme_config['plot_bgcolor'],
                                    paper_bgcolor=theme_config['paper_bgcolor'],
                                    font=dict(color=theme_config['font_color'])
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            except Exception as e:
                                st.error(f"Error generating normalized metric comparison plot: {e}")
                    
                    adv_tab_idx += 1
                
                # Time Decomposition
                if "Time Decomposition" in adv_tab_names:
                    with adv_tabs[adv_tab_idx]:
                        ts_col = st.selectbox(
                            "Select metric to decompose", 
                            num_cols, 
                            key="ts_col_select"
                        ) 
                        
                        try:
                            date_col = get_date_column(df)
                            ts_df = df.set_index(date_col)[ts_col].resample('D').sum().ffill()
                            
                            decomposition = seasonal_decompose(ts_df, model='additive', period=7)
                            
                            fig = make_subplots(
                                rows=4, 
                                cols=1, 
                                shared_xaxes=True,
                                subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=ts_df.index, 
                                    y=ts_df, 
                                    name='Observed',
                                    line=dict(color=theme_config['primary_color'])
                                ), 
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=decomposition.trend.index, 
                                    y=decomposition.trend, 
                                    name='Trend',
                                    line=dict(color=theme_config['secondary_color'])
                                ), 
                                row=2, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=decomposition.seasonal.index, 
                                    y=decomposition.seasonal, 
                                    name='Seasonal',
                                    line=dict(color=theme_config['accent_color'])
                                ), 
                                row=3, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=decomposition.resid.index, 
                                    y=decomposition.resid, 
                                    name='Residual',
                                    line=dict(color='#f39c12')
                                ), 
                                row=4, col=1
                            )
                            
                            fig.update_layout(
                                height=600, 
                                title_text="Time Series Decomposition",
                                showlegend=False,
                                plot_bgcolor=theme_config['plot_bgcolor'],
                                paper_bgcolor=theme_config['paper_bgcolor'],
                                font=dict(color=theme_config['font_color'])
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.warning(f"Couldn't decompose: {str(e)}")
                    
                    adv_tab_idx += 1
                
                # Composition Over Time
                if "Composition Over Time" in adv_tab_names:
                    with adv_tabs[adv_tab_idx]:
                        comp_col = st.selectbox(
                            "Select category", 
                            cat_cols, 
                            key="time_comp_col_select"
                        )
                        
                        metric_col = st.selectbox(
                            "Select metric", 
                            num_cols, 
                            key="time_comp_metric_select"
                        )
                        
                        comp_df = df.groupby([date_col, comp_col])[metric_col].sum().unstack().fillna(0)
                        
                        fig = px.area(
                            comp_df, 
                            title=f"{metric_col} Composition by {comp_col} Over Time",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title=metric_col,
                            legend_title=comp_col,
                            plot_bgcolor=theme_config['plot_bgcolor'],
                            paper_bgcolor=theme_config['paper_bgcolor'],
                            font=dict(color=theme_config['font_color'])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    adv_tab_idx += 1
                
                # 3D Visualization
                if "3D Visualization" in adv_tab_names:
                    with adv_tabs[adv_tab_idx]:
                        col3d1, col3d2 = st.columns([1, 3])
                        
                        with col3d1:
                            x_col = st.selectbox(
                                "X axis", 
                                num_cols, 
                                key="3d_x_col"
                            )
                            
                            y_col = st.selectbox(
                                "Y axis", 
                                [c for c in num_cols if c != x_col], 
                                key="3d_y_col"
                            )
                            
                            z_col = st.selectbox(
                                "Z axis", 
                                [c for c in num_cols if c not in [x_col, y_col]], 
                                key="3d_z_col"
                            )
                            
                            if cat_cols:
                                color_col = st.selectbox(
                                    "Color by", 
                                    ['None'] + cat_cols, 
                                    key="3d_color_col"
                                )
                                color_col = None if color_col == 'None' else color_col
                        
                        with col3d2:
                            fig = px.scatter_3d(
                                df, 
                                x=x_col, 
                                y=y_col, 
                                z=z_col, 
                                color=color_col,
                                title=f"3D Visualization of {x_col}, {y_col}, {z_col}",
                                opacity=0.7
                            )
                            
                            fig.update_layout(
                                margin=dict(l=0, r=0, b=0, t=30),
                                height=600,
                                plot_bgcolor=theme_config['plot_bgcolor'],
                                paper_bgcolor=theme_config['paper_bgcolor'],
                                font=dict(color=theme_config['font_color'])
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    adv_tab_idx += 1
                
                # Sunburst Chart
                if "Sunburst Chart" in adv_tab_names:
                    with adv_tabs[adv_tab_idx]:
                        path_cols = st.multiselect(
                            "Select hierarchy path", 
                            cat_cols, 
                            cat_cols[:2],
                            key="sunburst_path_select"
                        )
                        
                        if path_cols:
                            fig = px.sunburst(
                                df, 
                                path=path_cols, 
                                values=sales_metric,
                                title=f"Hierarchical View of {sales_metric}",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig.update_layout(
                                plot_bgcolor=theme_config['plot_bgcolor'],
                                paper_bgcolor=theme_config['paper_bgcolor'],
                                font=dict(color=theme_config['font_color'])
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

def main():
    # Get theme configuration
    theme_config = get_theme_config()
    
    # Apply professional theme
    apply_professional_theme(theme_config)
    
    # Set up the page configuration
    st.set_page_config(
        page_title="Professional EDA Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display the header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    with col2:
        st.title("Professional Exploratory Data Analysis")
        st.markdown("""
        <style>
            .title {
                color: #2c3e50;
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: #7f8c8d;
                font-size: 1.1rem;
                margin-top: 0;
            }
        </style>
        <h1 class="title">Professional Exploratory Data Analysis</h1>
        <p class="subtitle">Comprehensive data exploration and visualization tool</p>
        """, unsafe_allow_html=True)
    
    # Add a divider
    st.markdown("---")
    
    # File uploader in sidebar
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Upload your data file or use the default Coronation Bakery dataset"
        )
        
        st.markdown("---")
        st.header("About")
        st.markdown("""
        This professional EDA dashboard provides:
        - Automated data profiling
        - Interactive visualizations
        - Advanced analytics
        - Export capabilities
        
        [GitHub Repository](#)
        """)
    
    # Load the data
    with st.spinner("Loading and analyzing data..."):
        df = load_data(uploaded_file)
        
        if df is None:
            st.error("❌ Failed to load dataset. Please check your file and try again.")
            return
        
        # Convert date columns
        df = convert_to_datetime(df)
        
        # Detect column types
        col_types = detect_column_types(df)
        
        # Apply filters
        df = create_filters(df, col_types, theme_config)
    
    # Show dataset info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("Dataset Info")
        st.markdown(f"""
        - **Rows**: {len(df):,}
        - **Columns**: {len(df.columns):,}
        - **Size**: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB
        """)
        
        if len(col_types['numeric']) > 0:
            st.markdown("**Numeric Columns**:")
            for col in col_types['numeric'][:3]:
                st.markdown(f"- {col}")
            if len(col_types['numeric']) > 3:
                with st.expander("Show all numeric columns"):
                    for col in col_types['numeric'][3:]:
                        st.markdown(f"- {col}")
        
        if len(col_types['categorical']) > 0:
            st.markdown("**Categorical Columns**:")
            for col in col_types['categorical'][:3]:
                st.markdown(f"- {col}")
            if len(col_types['categorical']) > 3:
                with st.expander("Show all categorical columns"):
                    for col in col_types['categorical'][3:]:
                        st.markdown(f"- {col}")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Overview", 
        "📊 Distributions", 
        "⏳ Time Series", 
        "🔍 Missing Values", 
        "📈 Correlations", 
        "✨ Advanced Analytics"
    ])
    
    with tab1:
        # Overview section
        st.header("Dataset Overview")
        
        # Show KPIs
        show_kpi_cards(generate_kpis(df, col_types))
        
        # Data preview
        show_data_preview(df)
    
    with tab2:
        # Distributions section
        show_distributions(df, col_types)
    
    with tab3:
        # Time series section
        show_time_series(df, col_types)
    
    with tab4:
        # Missing values section
        show_missing_values(df)
    
    with tab5:
        # Correlations section
        show_correlations(df, col_types)
    
    with tab6:
        # Advanced analytics section
        show_correlation_visualizations(df, col_types)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>Professional EDA Dashboard • Built with Streamlit</p>
        <p>© 2023 Data Analytics Team • Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
