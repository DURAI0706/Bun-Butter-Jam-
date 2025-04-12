import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet  
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error

# Initialize session state variables
def initialize_session_state():
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ["Random Forest", "XGBoost"]
    if 'model_comparison_index' not in st.session_state:
        st.session_state.model_comparison_index = 0
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

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


@st.cache_data
def train_models_with_features(_X, _y, test_size, selected_models):
    """Train models using only the selected features with caching"""
    models_config = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "type": "sklearn"
        },
        "XGBoost": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            "type": "sklearn"
        },
        "Lasso": {
            "model": Lasso(random_state=42),
            "params": {"alpha": [0.01, 0.1, 1]},
            "type": "sklearn"
        },
        "SVR": {
            "model": SVR(),
            "params": {"C": [0.1, 1, 10], "kernel": ['linear', 'rbf']},
            "type": "sklearn"
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "params": {"n_neighbors": [3, 5, 10]},
            "type": "sklearn"
        },
        "ARIMA": {
            "model": None,
            "params": {"order": [(1, 1, 0), (2, 1, 1), (3, 1, 2)]},
            "type": "statsmodels"
        },
        "Holt-Winters": {
            "model": None,
            "params": {"trend": ['add', 'mul'], "seasonal": ['add', 'mul'], "seasonal_periods": [12]},
            "type": "statsmodels"
        },
        "Prophet": {
            "model": None,
            "params": {"seasonality_mode": ['additive', 'multiplicative'], 
                      "weekly_seasonality": [True], "daily_seasonality": [False]},
            "type": "prophet"
        }
    }

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=test_size, random_state=42)
    
    # Time series split for evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    
    results = {}
    best_models = {}

    for name in selected_models:
        if name not in models_config:
            st.warning(f"Model {name} not found in configuration")
            continue
            
        config = models_config[name]
        model_type = config["type"]
        
        try:
            if model_type == "sklearn":
                # Standard sklearn models
                grid_search = GridSearchCV(
                    config["model"], 
                    config["params"], 
                    scoring='r2', 
                    cv=tscv, 
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                
                results[name] = {
                    'MSE': mean_squared_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'Explained Variance': explained_variance_score(y_test, y_pred),
                    'Max Error': max_error(y_test, y_pred),
                    'Predictions': y_pred,
                    'Best Params': grid_search.best_params_
                }
                best_models[name] = best_model
                
            elif model_type == "statsmodels":
                # Time series models from statsmodels
                best_params = {}
                best_mse = float('inf')
                best_model = None
                best_preds = None
                
                # Convert to pandas Series for time series models
                # For time series models, we use the full series for training
                # and forecast the test portion
                full_series = pd.Series(_y)
                train_series = pd.Series(y_train)
                
                if name == "ARIMA":
                    for order in config["params"]["order"]:
                        try:
                            # Create and fit ARIMA model
                            model = ARIMA(train_series, order=order)
                            fitted = model.fit()
                            
                            # Forecast test data points
                            preds = fitted.forecast(steps=len(y_test))
                            mse = mean_squared_error(y_test, preds)
                            
                            if mse < best_mse:
                                best_mse = mse
                                best_model = fitted
                                best_preds = preds
                                best_params = {"order": order}
                        except Exception as e:
                            st.warning(f"ARIMA {order} failed: {str(e)}")
                            continue
                                
                elif name == "Holt-Winters":
                    for trend in config["params"]["trend"]:
                        for seasonal in config["params"]["seasonal"]:
                            for seasonal_periods in config["params"]["seasonal_periods"]:
                                try:
                                    # Create and fit Holt-Winters model
                                    model = ExponentialSmoothing(
                                        train_series, 
                                        trend=trend,
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods
                                    )
                                    fitted = model.fit()
                                    
                                    # Forecast test data points
                                    preds = fitted.forecast(steps=len(y_test))
                                    mse = mean_squared_error(y_test, preds)
                                    
                                    if mse < best_mse:
                                        best_mse = mse
                                        best_model = fitted
                                        best_preds = preds
                                        best_params = {
                                            "trend": trend,
                                            "seasonal": seasonal,
                                            "seasonal_periods": seasonal_periods
                                        }
                                except Exception as e:
                                    st.warning(f"Holt-Winters {trend}, {seasonal} failed: {str(e)}")
                                    continue
                
                # Calculate metrics if a model was successfully fit
                if best_model is not None:
                    try:
                        r2 = r2_score(y_test, best_preds)
                        mae = mean_absolute_error(y_test, best_preds)
                        explained_var = explained_variance_score(y_test, best_preds)
                        max_err = max_error(y_test, best_preds)
                        
                        results[name] = {
                            'MSE': best_mse,
                            'R2': r2,
                            'MAE': mae,
                            'Explained Variance': explained_var,
                            'Max Error': max_err,
                            'Predictions': best_preds,
                            'Best Params': best_params
                        }
                        best_models[name] = best_model
                    except Exception as e:
                        st.warning(f"Failed to calculate metrics for {name}: {str(e)}")
                else:
                    st.warning(f"Could not fit {name} model with any parameters")
                    
            elif model_type == "prophet":
                # Prophet model handling
                best_params = {}
                best_mse = float('inf')
                best_model = None
                best_preds = None
                
                # Create dataframes for Prophet
                # Prophet requires 'ds' (date) and 'y' (target) columns
                date_range = pd.date_range(start='2022-01-01', periods=len(_y))
                prophet_full = pd.DataFrame({
                    'ds': date_range,
                    'y': _y
                })
                
                # Split into train and test
                prophet_train = prophet_full.iloc[:len(y_train)]
                prophet_test_dates = prophet_full.iloc[len(y_train):]['ds']
                
                for seasonality_mode in config["params"]["seasonality_mode"]:
                    for weekly_seasonality in config["params"]["weekly_seasonality"]:
                        for daily_seasonality in config["params"]["daily_seasonality"]:
                            try:
                                # Create and fit Prophet model
                                model = Prophet(
                                    seasonality_mode=seasonality_mode,
                                    weekly_seasonality=weekly_seasonality,
                                    daily_seasonality=daily_seasonality
                                )
                                model.fit(prophet_train)
                                
                                # Create future dataframe for prediction
                                future = pd.DataFrame({'ds': prophet_test_dates})
                                forecast = model.predict(future)
                                preds = forecast['yhat'].values
                                
                                mse = mean_squared_error(y_test, preds)
                                
                                if mse < best_mse:
                                    best_mse = mse
                                    best_model = model
                                    best_preds = preds
                                    best_params = {
                                        "seasonality_mode": seasonality_mode,
                                        "weekly_seasonality": weekly_seasonality,
                                        "daily_seasonality": daily_seasonality
                                    }
                            except Exception as e:
                                st.warning(f"Prophet failed: {str(e)}")
                                continue
                
                # Calculate metrics if a model was successfully fit
                if best_model is not None:
                    try:
                        r2 = r2_score(y_test, best_preds)
                        mae = mean_absolute_error(y_test, best_preds)
                        explained_var = explained_variance_score(y_test, best_preds)
                        max_err = max_error(y_test, best_preds)
                        
                        results[name] = {
                            'MSE': best_mse,
                            'R2': r2,
                            'MAE': mae,
                            'Explained Variance': explained_var,
                            'Max Error': max_err,
                            'Predictions': best_preds,
                            'Best Params': best_params
                        }
                        best_models[name] = best_model
                    except Exception as e:
                        st.warning(f"Failed to calculate metrics for {name}: {str(e)}")
                else:
                    st.warning(f"Could not fit {name} model with any parameters")
        
        except Exception as e:
            st.error(f"Error training {name} model: {str(e)}")
    
    # Check if any models were successfully trained
    if not results:
        st.error("No models were successfully trained. Please try different features or models.")
    
    return results, best_models, X_test, y_test

def create_model_comparison_plots(results, y_test):
    """Create all model comparison visualizations"""
    # Metrics comparison
    metrics_df = pd.DataFrame.from_dict({
        name: {
            'MSE': res['MSE'],
            'R²': res['R2'],
            'MAE': res['MAE'],
            'Explained Variance': res['Explained Variance']
        }
        for name, res in results.items()
    }, orient='index')
    
    # Display metrics table
    st.dataframe(
        metrics_df.style
        .background_gradient(cmap='Blues', subset=['R²'])
        .format({'MSE': '{:.2f}', 'R²': '{:.3f}', 'MAE': '{:.2f}', 'Explained Variance': '{:.3f}'})
    )
    
    # R² comparison chart
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
    
    # Error metrics comparison
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
    
    # Best parameters
    st.subheader("Best Parameters")
    for name, res in results.items():
        st.write(f"**{name}**: {res['Best Params']}")

def create_model_prediction_plots(results, y_test, selected_model, target_variable):
    """Create prediction visualizations for a specific model
    
    Args:
        results: Dictionary containing model results
        y_test: Actual target values
        selected_model: Name of the model to visualize
        target_variable: Name of the target variable for labeling
    """
    # Actual vs Predicted scatter plot
    actual_vs_pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': results[selected_model]['Predictions']
    })
    
    fig_scatter = px.scatter(
        actual_vs_pred_df,
        x='Actual',
        y='Predicted',  # Fixed typo here (was 'Predicted')
        title=f"{selected_model}: Actual vs Predicted Values",
        opacity=0.7
    )
    
    # Add perfect prediction line
    max_val = max(actual_vs_pred_df['Actual'].max(), actual_vs_pred_df['Predicted'].max())
    min_val = min(actual_vs_pred_df['Actual'].min(), actual_vs_pred_df['Predicted'].min())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Time series prediction plot
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        y=y_test,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    fig_pred.add_trace(go.Scatter(
        y=results[selected_model]['Predictions'],
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dot')
    ))
    fig_pred.update_layout(
        xaxis_title="Observation",
        yaxis_title=target_variable,
        hovermode="x unified"
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Error distribution
    errors = y_test - results[selected_model]['Predictions']
    fig_error_dist = px.histogram(
        errors,
        title="Prediction Error Distribution",
        labels={'value': 'Error'},
        marginal='box'
    )
    st.plotly_chart(fig_error_dist, use_container_width=True)

def main():
    st.title("🧁 Coronation Bakery Sales Analytics")
    initialize_session_state()
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own dataset (optional)", 
        type=["csv", "xlsx"],
        help="Upload your bakery sales data in CSV or Excel format"
    )
    
    # Load data with error handling
    try:
        df = load_data(uploaded_file)
        if df is None:
            st.error("Failed to load data. Please check your file format.")
            return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    original_df = df.copy()
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Step 1: Select target variable
        target_options = get_valid_targets(df)
        if not target_options:
            st.error("No valid target variables found in the dataset")
            return
            
        target_variable = st.selectbox(
            "1. Select Target Variable:",
            options=target_options,
            index=target_options.index('Daily_Total') if 'Daily_Total' in target_options else 0,
            help="Choose the variable you want to predict"
        )
        
        # Step 2: Test size selection
        st.subheader("2. Test Set Size")
        test_size = st.slider(
            "Select test set size (%):",
            min_value=10,
            max_value=30,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        ) / 100
        
        # Step 3: Feature Selection
        st.subheader("3. Feature Selection")
        num_features = st.slider(
            "Number of top features to select:",
            min_value=5,
            max_value=30,
            value=10,
            step=1
        )

        # Get initial feature selection
        df_processed = preprocess_data(df, target_variable)
        X = df_processed.drop(columns=[target_variable])
        y = df_processed[target_variable]

        # Initialize session state for selected features if it doesn't exist
        if 'selected_features' not in st.session_state:
            selector = SelectKBest(score_func=f_regression, k=num_features)
            selector.fit(X, y)
            st.session_state.selected_features = X.columns[selector.get_support()].tolist()
        # Update only if slider value changes
        elif 'prev_num_features' not in st.session_state or st.session_state.prev_num_features != num_features:
            selector = SelectKBest(score_func=f_regression, k=num_features)
            selector.fit(X, y)
            st.session_state.selected_features = X.columns[selector.get_support()].tolist()
            st.session_state.prev_num_features = num_features

        # Feature selection multiselect
        selector = SelectKBest(score_func=f_classif, k=num_features)
        selector.fit(X, y)
        selected_mask = selector.get_support()
        top_features = X.columns[selected_mask].tolist()
        
        # Now show multiselect for the top features
        selected_features = st.multiselect(
            "Select features for training",
            options=top_features,
            default=top_features
        )

        # Update the session state with user's selection
        st.session_state.selected_features = selected_features
        
        # Step 4: Model selection
        st.subheader("4. Select Models")
        model_options = ["Random Forest", "XGBoost", "Lasso", "SVR", "KNN","ARIMA","Holt-Winters","Prophet"]
        st.session_state.selected_models = st.multiselect(
            "Choose models to run:",
            options=model_options,
            default=st.session_state.selected_models,
            help="Select which machine learning models to compare"
        )
        
        # Step 5: Run Analysis
        st.subheader("5. Run Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Run Models", type="primary", help="Train models with selected configuration"):
                if not st.session_state.selected_models:
                    st.error("Please select at least one model")
                    return
                if not st.session_state.selected_features:
                    st.error("Please select at least one feature")
                    return
                    
                with st.spinner("Training models... This may take a few minutes"):
                    try:
                        # Use only the user-selected features
                        X_selected = X[st.session_state.selected_features]
                        
                        # Train models with caching
                        results, models, X_test, y_test = train_models_with_features(
                            X_selected, y, test_size, st.session_state.selected_models
                        )
                        
                        # Store results in session state
                        st.session_state.trained_models = {
                            'results': results,
                            'models': models,
                            'X_test': X_test,
                            'y_test': y_test,
                            'target_variable': target_variable,
                            'df_processed': df_processed
                        }
                        st.session_state.analysis_complete = True
                        st.session_state.models_trained_once = True  # New flag
                        st.success("✅ Model training completed!")
                    except Exception as e:
                        st.error(f"Model training failed: {str(e)}")
        
        # Add a rerun button if models have been trained once
        with col2:
            if st.session_state.get('models_trained_once', False):
                if st.button("🔄 Retrain Models", help="Run the models again with current configuration"):
                    st.session_state.analysis_complete = False  # Reset to trigger training
                    st.rerun()  # This will rerun the script and hit the training code again
                    
    # Data Overview Section
    st.header("📊 Data Overview")
    data_tabs = st.tabs(["Time Patterns", "Product Performance", "Seasonal Trends"])
    
    with data_tabs[0]:
        st.subheader("Daily Sales Patterns")
        
        # Prepare data for visualization
        if 'DayOfWeek' in original_df.columns and target_variable in original_df.columns:
            # Map day of week numbers to names
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
            
            # Add average weekend vs weekday comparison
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
                # Calculate rolling average to show trend
                if 'Rolling_7' in original_df.columns:
                    # Create time series with the first day of each rolling window
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
        
        # Check if relevant columns exist
        if 'Product_Type' in original_df.columns:
            product_data = original_df.groupby('Product_Type')[target_variable].sum().reset_index()
            product_data = product_data.sort_values(target_variable, ascending=False)
            
            # Top products by total sales
            fig_product = px.pie(
                product_data,
                values=target_variable,
                names='Product_Type',
                title=f'Product Mix by {target_variable}',
                hole=0.4,
            )
            st.plotly_chart(fig_product, use_container_width=True)
            
            # Product performance comparison
            fig_product_bar = px.bar(
                product_data.head(10),  # Top 10 products
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
            # Map month numbers to names
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
            
            # Add range slider to zoom in on specific time periods
            fig_month.update_layout(
                xaxis=dict(
                    type='category'
                )
            )
            st.plotly_chart(fig_month, use_container_width=True)
            
            # Add quarter analysis if we have enough data
            if len(month_data) >= 4:
                # Create quarter column
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
    
    # Main content area
    if st.session_state.analysis_complete:
        results = st.session_state.trained_models['results']
        y_test = st.session_state.trained_models['y_test']
        target_variable = st.session_state.trained_models['target_variable']
        df_processed = st.session_state.trained_models['df_processed']
        
        st.header("🔍 Model Analysis")
        model_tabs = st.tabs(["📊 Performance Metrics", "🔮 Predictions", "📈 Feature Analysis"])
        
        with model_tabs[0]:
            create_model_comparison_plots(results, y_test)
            
        with model_tabs[1]:
            st.subheader("Actual vs Predicted")
            
            if results:  # Check if results dictionary is not empty
                # Create dropdown to select model
                selected_model = st.selectbox(
                    "Select model to visualize:", 
                    options=list(results.keys()),
                    index=st.session_state.model_comparison_index,
                    key="model_selector"
                )
                
                # Update session state index when selection changes
                current_index = list(results.keys()).index(selected_model)
                if current_index != st.session_state.model_comparison_index:
                    st.session_state.model_comparison_index = current_index
                    
                create_model_prediction_plots(
                    results, 
                    y_test, 
                    selected_model, 
                    target_variable
                )
            else:
                st.warning("No models were successfully trained. Please try different features or models.")
            
        with model_tabs[2]:
            st.subheader("Feature Analysis")
            
            # Correlation heatmap
            st.subheader("📈 Correlation Heatmap (Selected Features)")
            selected_cols = st.session_state.selected_features.copy()
            if target_variable not in selected_cols:
                selected_cols.append(target_variable)
                
            corr_matrix = df_processed[selected_cols].corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale='viridis',
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Feature importance for the best model
            best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
            best_model = st.session_state.trained_models['models'][best_model_name]
            
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
                        title='Feature Importance Scores',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Get top important features for bivariate analysis
                    top_features = feature_importance.head(5)['Feature'].tolist()
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
                    # Fallback to correlation
                    corr_with_target = df_processed[st.session_state.selected_features].corrwith(
                        df_processed[target_variable]
                    )
                    top_features = corr_with_target.abs().sort_values(ascending=False).head(5).index.tolist()
            else:
                # Use correlation if feature importance isn't available
                corr_with_target = df_processed[st.session_state.selected_features].corrwith(
                    df_processed[target_variable]
                )
                top_features = corr_with_target.abs().sort_values(ascending=False).head(5).index.tolist()
            
            # Bivariate Analysis Section
            st.subheader("🔍 Bivariate Analysis")
            st.write("Explore relationships between features and the target variable")
            
            selected_feature = st.selectbox(
                "Select feature to analyze:",
                options=top_features,
                key="bivariate_feature_selector"
            )
            
            # Create analysis columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot with trendline
                fig_scatter = px.scatter(
                    df_processed,
                    x=selected_feature,
                    y=target_variable,
                    title=f'{selected_feature} vs {target_variable}',
                    trendline='ols',
                    trendline_color_override='red',
                    opacity=0.6
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Box plot for categorical features
                if df_processed[selected_feature].nunique() < 10:
                    fig_box = px.box(
                        df_processed,
                        x=selected_feature,
                        y=target_variable,
                        title=f'Distribution by {selected_feature}'
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Density plot for numeric features
                if pd.api.types.is_numeric_dtype(df_processed[selected_feature]):
                    if len(df_processed) > 1000:
                        fig_density = px.density_heatmap(
                            df_processed,
                            x=selected_feature,
                            y=target_variable,
                            title='Data Density',
                            nbinsx=20,
                            nbinsy=20,
                            color_continuous_scale='Viridis'
                        )
                    else:
                        fig_density = px.violin(
                            df_processed,
                            x=selected_feature,
                            y=target_variable,
                            title='Distribution',
                            box=True
                        )
                    st.plotly_chart(fig_density, use_container_width=True)
                
                # Correlation stats
                corr = df_processed[selected_feature].corr(df_processed[target_variable])
                st.metric(
                    label=f"Correlation with {target_variable}",
                    value=f"{corr:.3f}",
                    help="Pearson correlation coefficient (-1 to 1)"
                )


# This code was causing the error - removed it from here
# It was checking 'results' before it was defined
# if not results:
#    st.error("No models were successfully trained. Please try different features or models.")
#    return {}, {}, X_test, y_test

if __name__ == "__main__":
    main()
