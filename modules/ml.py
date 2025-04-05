import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

@st.cache_data
def load_data(uploaded_file=None):
    """Loads default or user-uploaded dataset with feature engineering"""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv("Coronation Bakery Dataset.csv")

        # Ensure 'Date' column is datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Feature Engineering
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        df['Lag_1'] = df['Daily_Items_Sold'].shift(1).fillna(0)
        df['Lag_7'] = df['Daily_Items_Sold'].shift(7).fillna(0)
        df['Lag_30'] = df['Daily_Items_Sold'].shift(30).fillna(0)

        df['Rolling_7'] = df['Daily_Items_Sold'].rolling(window=7, min_periods=1).mean()
        df['Rolling_30'] = df['Daily_Items_Sold'].rolling(window=30, min_periods=1).mean()

        df['EWMA_7'] = df['Daily_Items_Sold'].ewm(span=7, min_periods=1).mean()

        df.drop(columns=['Date'], inplace=True)
        df = df.dropna(subset=['Daily_Total', 'Daily_Items_Sold'])

        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {str(e)}")
        return None

def get_valid_targets(df):
    """Returns list of valid target variables (excluding engineered features)"""
    original_columns = ['Quantity', 'Price_Per_Unit', 'Total_Amount', 
                       'Daily_Total', 'Daily_Items_Sold']
    return [col for col in original_columns if col in df.columns and df[col].dtype in ['int64', 'float64']]

def preprocess_data(df, target_col):
    categorical_cols = [col for col in ['Seller_Name', 'Product_Type'] if col in df.columns]
    
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        categorical_encoded = encoder.fit_transform(df[categorical_cols])
        categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out())
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, categorical_df], axis=1)
    
    return df

def train_models_with_features(X, y, test_size, selected_models):
    """Train models using only the selected features."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models_config = {
        "Random Forest": {"model": RandomForestRegressor(random_state=42), "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}},
        "XGBoost": {"model": GradientBoostingRegressor(random_state=42), "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}},
        "Lasso": {"model": Lasso(random_state=42), "params": {"alpha": [0.01, 0.1, 1]}},
        "SVR": {"model": SVR(), "params": {"C": [0.1, 1, 10], "kernel": ['linear', 'rbf']}},
        "KNN": {"model": KNeighborsRegressor(), "params": {"n_neighbors": [3, 5, 10]}}
    }

    results = {}
    best_models = {}

    for name in selected_models:
        config = models_config[name]
        grid_search = GridSearchCV(
            config["model"], 
            config["params"], 
            scoring='r2', 
            cv=3, 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        results[name] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'Predictions': y_pred,
            'Best Params': grid_search.best_params_
        }
        best_models[name] = best_model

    return results, best_models, X_test, y_test

def main():
    st.title("🧁 Coronation Bakery Sales Analytics")
    
    # Initialize session state variables
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ["Random Forest", "XGBoost"]
    if 'prev_num_features' not in st.session_state:
        st.session_state.prev_num_features = 10

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own dataset (optional)", type=["csv", "xlsx"]
    )

    # Load data
    df = load_data(uploaded_file)
    if df is None:
        return

    original_df = df.copy()

    # Feedback message
    if uploaded_file:
        st.sidebar.success(f"✅ Using uploaded file: {uploaded_file.name}")
    else:
        st.sidebar.info("📂 Using default: Coronation Bakery Dataset.csv")

    # Configuration sidebar
    with st.sidebar:
        st.header("Model Configuration")

        # Step 1: Select target variable
        target_variable = st.selectbox(
            "1. Select Target Variable:",
            options=get_valid_targets(df),
            index=get_valid_targets(df).index('Daily_Total') if 'Daily_Total' in get_valid_targets(df) else 0,
            key='target_variable'
        )

        # Step 2: Test size selection
        st.subheader("2. Test Set Size")
        test_size = st.slider(
            "Select test set size (%):",
            min_value=10,
            max_value=30,
            value=20,
            step=5,
            key='test_size'
        ) / 100

        st.subheader("3. Feature Selection")
        num_features = st.slider(
            "Number of top features to select:",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key='num_features'
        )

        # Get initial feature selection
        df_processed = preprocess_data(df, target_variable)
        X = df_processed.drop(columns=[target_variable])
        y = df_processed[target_variable]

        # Update feature selection if needed
        if 'prev_num_features' not in st.session_state or st.session_state.prev_num_features != num_features:
            selector = SelectKBest(score_func=f_regression, k=num_features)
            selector.fit(X, y)
            st.session_state.selected_features = X.columns[selector.get_support()].tolist()
            st.session_state.prev_num_features = num_features

        # Feature selection multiselect
        selected_features = st.multiselect(
            "Select features for training",
            options=st.session_state.selected_features,
            default=st.session_state.selected_features,
            key='feature_selector'
        )
        st.session_state.selected_features = selected_features

        # Step 4: Model selection
        st.subheader("4. Select Models")
        model_options = ["Random Forest", "XGBoost", "Lasso", "SVR", "KNN"]
        st.session_state.selected_models = st.multiselect(
            "Choose models to run:",
            options=model_options,
            default=st.session_state.selected_models,
            key='model_selector'
        )

        # Step 5: Run Analysis
        st.subheader("5. Run Analysis")
        run_button = st.button("Run Models", type="primary", key='run_button')
    
    # Data Overview Section
    st.header("📊 Data Overview")
    data_tabs = st.tabs(["Time Patterns", "Product Performance", "Seasonal Trends"])
    
    with data_tabs[0]:
        st.subheader("Daily Sales Patterns")
        
        if 'DayOfWeek' in original_df.columns and target_variable in original_df.columns:
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                         4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            day_pattern_data = original_df.groupby('DayOfWeek')[target_variable].mean().reset_index()
            day_pattern_data['Day'] = day_pattern_data['DayOfWeek'].map(day_names)
            day_pattern_data = day_pattern_data.sort_values('DayOfWeek')
            
            fig_day_pattern = px.bar(
                day_pattern_data,
                x='Day',
                y=target_variable,
                title=f'Average {target_variable} by Day of Week',
                color=target_variable,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_day_pattern, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                weekend_avg = original_df[original_df['IsWeekend'] == 1][target_variable].mean()
                weekday_avg = original_df[original_df['IsWeekend'] == 0][target_variable].mean()
                weekend_data = pd.DataFrame({
                    'Day Type': ['Weekday', 'Weekend'],
                    target_variable: [weekday_avg, weekend_avg]
                })
                
                fig_weekend = px.bar(
                    weekend_data,
                    x='Day Type',
                    y=target_variable,
                    title=f'Weekday vs Weekend {target_variable}',
                    color='Day Type',
                    color_discrete_map={'Weekday': 'lightblue', 'Weekend': 'darkblue'}
                )
                st.plotly_chart(fig_weekend, use_container_width=True)
            
            with col2:
                if 'Rolling_7' in original_df.columns:
                    rolling_data = original_df[['Rolling_7']].copy()
                    rolling_data = rolling_data.reset_index().rename(columns={'index': 'Day'})
                    
                    fig_rolling = px.line(
                        rolling_data,
                        x='Day',
                        y='Rolling_7',
                        title='7-Day Rolling Average Sales',
                        labels={'Rolling_7': 'Average Sales'}
                    )
                    st.plotly_chart(fig_rolling, use_container_width=True)
    
    with data_tabs[1]:
        st.subheader("Product Performance Analysis")
        
        if 'Product_Type' in original_df.columns:
            product_data = original_df.groupby('Product_Type')[target_variable].sum().reset_index()
            product_data = product_data.sort_values(target_variable, ascending=False)
            
            fig_product = px.pie(
                product_data,
                values=target_variable,
                names='Product_Type',
                title=f'Product Mix by {target_variable}',
                hole=0.4,
            )
            st.plotly_chart(fig_product, use_container_width=True)
            
            fig_product_bar = px.bar(
                product_data.head(10),
                x='Product_Type',
                y=target_variable,
                title=f'Top Products by {target_variable}',
                color=target_variable,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_product_bar, use_container_width=True)
    
    with data_tabs[2]:
        st.subheader("Seasonal Analysis")
        
        if 'Month' in original_df.columns and target_variable in original_df.columns:
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                           7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            
            month_data = original_df.groupby('Month')[target_variable].mean().reset_index()
            month_data['Month_Name'] = month_data['Month'].map(month_names)
            month_data = month_data.sort_values('Month')
            
            fig_month = px.line(
                month_data,
                x='Month_Name',
                y=target_variable,
                title=f'Monthly {target_variable} Trends',
                markers=True
            )
            
            fig_month.update_layout(xaxis=dict(type='category'))
            st.plotly_chart(fig_month, use_container_width=True)
            
            if len(month_data) >= 4:
                month_data['Quarter'] = ((month_data['Month'] - 1) // 3) + 1
                quarter_data = month_data.groupby('Quarter')[target_variable].mean().reset_index()
                
                fig_quarter = px.bar(
                    quarter_data,
                    x='Quarter',
                    y=target_variable,
                    title=f'Quarterly {target_variable} Analysis',
                    color=target_variable,
                    text=round(quarter_data[target_variable], 2)
                )
                fig_quarter.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig_quarter, use_container_width=True)
    
    # Model training when button is clicked
    if run_button:
        if not st.session_state.selected_models:
            st.error("Please select at least one model to run")
        elif not st.session_state.selected_features:
            st.error("Please select at least one feature")
        else:
            with st.spinner("Preprocessing data..."):
                df_processed = preprocess_data(df, target_variable)
                X = df_processed[st.session_state.selected_features]
                y = df_processed[target_variable]
            
            with st.spinner(f"Training {len(st.session_state.selected_models)} models..."):
                results, models, X_test, y_test = train_models_with_features(
                    X, y, test_size, st.session_state.selected_models
                )
                st.session_state.results = results
                st.session_state.models = models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.model_trained = True
            st.success("Model training completed!")
    
    # Display model results
    if st.session_state.model_trained and st.session_state.results is not None:
        st.header("🔍 Model Analysis")
        model_tabs = st.tabs(["Performance Metrics", "Predictions", "Feature Analysis"])
        
        with model_tabs[0]:
            st.subheader("Model Performance Comparison")
            metrics_df = pd.DataFrame.from_dict({
                name: {
                    'MSE': res['MSE'],
                    'R²': res['R2'],
                    'MAE': res['MAE']
                }
                for name, res in st.session_state.results.items()
            }, orient='index')
            
            st.dataframe(
                metrics_df.style
                .background_gradient(cmap='Blues', subset=['R²'])
                .format({'MSE': '{:.2f}', 'R²': '{:.3f}', 'MAE': '{:.2f}'})
            )
            
            fig_r2 = px.bar(
                metrics_df.reset_index().rename(columns={'index': 'Model'}),
                x='Model',
                y='R²',
                title='Model R² Score Comparison',
                color='R²',
                color_continuous_scale='Viridis',
                text='R²'
            )
            fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_r2, use_container_width=True)
            
            error_df = metrics_df[['MSE', 'MAE']].reset_index().rename(columns={'index': 'Model'})
            error_df = pd.melt(error_df, id_vars=['Model'], value_vars=['MSE', 'MAE'], 
                              var_name='Metric', value_name='Value')
            
            fig_error = px.bar(
                error_df,
                x='Model',
                y='Value',
                color='Metric',
                barmode='group',
                title='Model Error Metrics Comparison',
                text='Value'
            )
            fig_error.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_error, use_container_width=True)
        
        with model_tabs[1]:
            st.subheader("Actual vs Predicted")
            
            selected_model = st.selectbox("Select model to visualize:", 
                                         options=list(st.session_state.results.keys()),
                                         key='model_visualization')
            
            actual_vs_pred_df = pd.DataFrame({
                'Actual': st.session_state.y_test,
                'Predicted': st.session_state.results[selected_model]['Predictions']
            })
            
            fig_scatter = px.scatter(
                actual_vs_pred_df,
                x='Actual',
                y='Predicted',
                title=f"{selected_model}: Actual vs Predicted Values",
                opacity=0.7
            )
            
            max_val = max(actual_vs_pred_df['Actual'].max(), actual_vs_pred_df['Predicted'].max())
            min_val = min(actual_vs_pred_df['Actual'].min(), actual_vs_pred_df['Predicted'].min())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                y=st.session_state.y_test,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            )
            
            fig_pred.add_trace(go.Scatter(
                y=st.session_state.results[selected_model]['Predictions'],
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dot')
            )
            
            fig_pred.update_layout(
                xaxis_title="Observation",
                yaxis_title=target_variable,
                hovermode="x unified"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            errors = st.session_state.y_test - st.session_state.results[selected_model]['Predictions']
            
            fig_error_dist = px.histogram(
                errors,
                title="Prediction Error Distribution",
                labels={'value': 'Error'},
                marginal='box'
            )
            st.plotly_chart(fig_error_dist, use_container_width=True)
        
        with model_tabs[2]:
            st.subheader("Feature Analysis")
            
            st.subheader("📈 Correlation Heatmap (Selected Features)")
            
            selected_cols = st.session_state.selected_features.copy()
            if target_variable not in selected_cols:
                selected_cols.append(target_variable)
                
            corr_matrix = df_processed[selected_cols].corr()
            
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            best_model_name = max(st.session_state.results.items(), key=lambda x: x[1]['R2'])[0]
            best_model = st.session_state.models[best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                try:
                    st.subheader(f"🧠 Feature Importance ({best_model_name})")
                    feature_importance = pd.DataFrame({
                        'Feature': st.session_state.selected_features,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance (Selected Features)',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    selected_feature = st.selectbox(
                        "Select feature for detailed analysis:",
                        options=feature_importance['Feature'].head(5).tolist(),
                        key='bivariate_feature'
                    )
                    
                    fig_scatter = px.scatter(
                        df_processed,
                        x=selected_feature,
                        y=target_variable,
                        title=f'{selected_feature} vs {target_variable}',
                        trendline='ols',
                        opacity=0.6
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
    else:
        st.info("Configure your models and click 'Run Models' to see results")

if __name__ == "__main__":
    main()
