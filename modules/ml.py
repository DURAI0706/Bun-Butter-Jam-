import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Configuration
st.set_page_config(layout="wide", page_title="🧁 Coronation Bakery Analytics")

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
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter

        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'Lag_{lag}'] = df['Daily_Items_Sold'].shift(lag).fillna(method='bfill')
        
        # Rolling features
        for window in [3, 7, 14, 30]:
            df[f'Rolling_Mean_{window}'] = df['Daily_Items_Sold'].rolling(window=window, min_periods=1).mean()
            df[f'Rolling_Std_{window}'] = df['Daily_Items_Sold'].rolling(window=window, min_periods=1).std()
        
        # Exponential moving averages
        for span in [7, 14, 30]:
            df[f'EWMA_{span}'] = df['Daily_Items_Sold'].ewm(span=span, min_periods=1).mean()
        
        # Seasonality features
        df['MonthSin'] = np.sin(2 * np.pi * df['Month']/12)
        df['MonthCos'] = np.cos(2 * np.pi * df['Month']/12)
        df['DayOfYearSin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
        df['DayOfYearCos'] = np.cos(2 * np.pi * df['DayOfYear']/365)

        # Store the date column separately before dropping
        dates = df['Date'].copy()
        df.drop(columns=['Date'], inplace=True)
        df = df.dropna(subset=['Daily_Total', 'Daily_Items_Sold'])

        return df, dates
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {str(e)}")
        return None, None

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

def train_time_series_models(dates, y, test_size, selected_models):
    """Train time series forecasting models"""
    # Create a DataFrame with dates and target
    ts_data = pd.DataFrame({'ds': dates, 'y': y})
    ts_data = ts_data.dropna()
    
    # Split into train and test
    test_size = int(len(ts_data) * test_size)
    train, test = ts_data.iloc[:-test_size], ts_data.iloc[-test_size:]
    
    results = {}
    models = {}
    
    # Model configurations
    ts_models_config = {
        "ARIMA": {
            "model": ARIMA,
            "params": {
                "order": [(1,0,0), (1,1,0), (2,1,0), (1,1,1)],
                "seasonal_order": [(0,0,0,0), (1,0,0,7), (0,1,0,7)]
            }
        },
        "SARIMA": {
            "model": SARIMAX,
            "params": {
                "order": [(1,0,0), (1,1,0), (2,1,0)],
                "seasonal_order": [(1,0,0,7), (0,1,0,7), (1,1,0,7)],
                "trend": ['n', 'c', 't', 'ct']
            }
        },
        "Exponential Smoothing": {
            "model": ExponentialSmoothing,
            "params": {
                "trend": ['add', 'mul', None],
                "seasonal": ['add', 'mul', None],
                "seasonal_periods": [7, 14, None]
            }
        },
        "Prophet": {
            "model": Prophet,
            "params": {
                "growth": ['linear', 'flat'],
                "seasonality_mode": ['additive', 'multiplicative'],
                "changepoint_prior_scale": [0.01, 0.1, 0.5],
                "seasonality_prior_scale": [0.1, 1.0, 10.0]
            }
        }
    }
    
    for name in selected_models:
        if name not in ts_models_config:
            continue
            
        try:
            if name == "Prophet":
                # Prophet has a different API
                model = Prophet(
                    growth=ts_models_config[name]["params"]["growth"][0],
                    seasonality_mode=ts_models_config[name]["params"]["seasonality_mode"][0],
                    changepoint_prior_scale=ts_models_config[name]["params"]["changepoint_prior_scale"][0],
                    seasonality_prior_scale=ts_models_config[name]["params"]["seasonality_prior_scale"][0]
                )
                model.fit(train)
                future = model.make_future_dataframe(periods=test_size)
                forecast = model.predict(future)
                y_pred = forecast['yhat'][-test_size:].values
            else:
                # For ARIMA/SARIMA/ETS
                best_score = float('inf')
                best_params = None
                best_model = None
                
                # Grid search over parameters
                for params in ts_models_config[name]["params"]:
                    try:
                        if name == "ARIMA":
                            model = ARIMA(train['y'], order=params["order"], seasonal_order=params["seasonal_order"])
                        elif name == "SARIMA":
                            model = SARIMAX(train['y'], order=params["order"], seasonal_order=params["seasonal_order"], trend=params["trend"])
                        elif name == "Exponential Smoothing":
                            model = ExponentialSmoothing(train['y'], trend=params["trend"], seasonal=params["seasonal"], seasonal_periods=params["seasonal_periods"])
                        
                        fit_model = model.fit()
                        y_pred = fit_model.forecast(test_size)
                        
                        # Evaluate on validation set (last 20% of train)
                        val_size = int(len(train) * 0.2)
                        val_train = train.iloc[:-val_size]
                        val_test = train.iloc[-val_size:]
                        
                        if name == "Prophet":
                            val_model = Prophet(
                                growth=params["growth"],
                                seasonality_mode=params["seasonality_mode"],
                                changepoint_prior_scale=params["changepoint_prior_scale"],
                                seasonality_prior_scale=params["seasonality_prior_scale"]
                            )
                            val_model.fit(val_train)
                            val_future = val_model.make_future_dataframe(periods=val_size)
                            val_forecast = val_model.predict(val_future)
                            val_pred = val_forecast['yhat'][-val_size:].values
                        else:
                            if name == "ARIMA":
                                val_model = ARIMA(val_train['y'], order=params["order"], seasonal_order=params["seasonal_order"])
                            elif name == "SARIMA":
                                val_model = SARIMAX(val_train['y'], order=params["order"], seasonal_order=params["seasonal_order"], trend=params["trend"])
                            elif name == "Exponential Smoothing":
                                val_model = ExponentialSmoothing(val_train['y'], trend=params["trend"], seasonal=params["seasonal"], seasonal_periods=params["seasonal_periods"])
                            
                            val_fit = val_model.fit()
                            val_pred = val_fit.forecast(val_size)
                        
                        score = mean_squared_error(val_test['y'], val_pred)
                        
                        if score < best_score:
                            best_score = score
                            best_params = params
                            best_model = fit_model if name != "Prophet" else model
                    except:
                        continue
                
                if best_model is not None:
                    if name != "Prophet":
                        y_pred = best_model.forecast(test_size)
            
            # Calculate metrics
            results[name] = {
                'MSE': mean_squared_error(test['y'], y_pred),
                'R2': r2_score(test['y'], y_pred),
                'MAE': mean_absolute_error(test['y'], y_pred),
                'Predictions': y_pred,
                'Best Params': best_params if name != "Prophet" else model.params
            }
            models[name] = best_model if name != "Prophet" else model
            
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
            continue
    
    return results, models, test['ds'], test['y']

def train_regression_models(X, y, test_size, selected_models):
    """Train regression models with time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=3)
    
    models_config = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.001, 0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        },
        "Lasso": {
            "model": Lasso(random_state=42),
            "params": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "selection": ['cyclic', 'random'],
                "max_iter": [1000, 5000, 10000]
            }
        },
        "Ridge": {
            "model": Ridge(random_state=42),
            "params": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        },
        "ElasticNet": {
            "model": ElasticNet(random_state=42),
            "params": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "selection": ['cyclic', 'random'],
                "max_iter": [1000, 5000, 10000]
            }
        },
        "SVR": {
            "model": SVR(),
            "params": {
                "C": [0.1, 1, 10, 100],
                "kernel": ['linear', 'rbf', 'poly', 'sigmoid'],
                "gamma": ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
                "epsilon": [0.01, 0.1, 0.5, 1.0]
            }
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "params": {
                "n_neighbors": [3, 5, 7, 10, 15, 20],
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                "leaf_size": [10, 20, 30, 50],
                "p": [1, 2]  # 1: manhattan, 2: euclidean
            }
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['auto', 'sqrt', 'log2'],
                "splitter": ['best', 'random']
            }
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0],
                "loss": ['linear', 'square', 'exponential']
            }
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        },
        "MLP": {
            "model": MLPRegressor(random_state=42, max_iter=1000),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50,50), (100,50)],
                "activation": ['identity', 'logistic', 'tanh', 'relu'],
                "solver": ['lbfgs', 'sgd', 'adam'],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ['constant', 'invscaling', 'adaptive'],
                "learning_rate_init": [0.001, 0.01, 0.1]
            }
        },
        "LightGBM": {
            "model": LGBMRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.001, 0.01, 0.1],
                "num_leaves": [31, 50, 100],
                "max_depth": [-1, 5, 10],
                "min_child_samples": [20, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        },
        "CatBoost": {
            "model": CatBoostRegressor(random_state=42, silent=True),
            "params": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.001, 0.01, 0.1],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1, 3, 5, 7],
                "border_count": [32, 64, 128]
            }
        }
    }
    
    results = {}
    best_models = {}
    
    for name in selected_models:
        if name not in models_config:
            continue
            
        config = models_config[name]
        grid_search = GridSearchCV(
            config["model"], 
            config["params"], 
            scoring='neg_mean_squared_error', 
            cv=tscv,
            n_jobs=-1,
            verbose=0
        )
        
        try:
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X[-test_size:])
            
            results[name] = {
                'MSE': mean_squared_error(y[-test_size:], y_pred),
                'R2': r2_score(y[-test_size:], y_pred),
                'MAE': mean_absolute_error(y[-test_size:], y_pred),
                'Predictions': y_pred,
                'Best Params': grid_search.best_params_
            }
            best_models[name] = best_model
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
            continue
    
    return results, best_models, X[-test_size:], y[-test_size:]

def main():
    st.title("🧁 Coronation Bakery Sales Analytics")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own dataset (optional)", type=["csv", "xlsx"]
    )
    
    # Load data
    df, dates = load_data(uploaded_file)
    if df is None:
        return
    
    original_df = df.copy()
    
    # Feedback message
    if uploaded_file:
        st.sidebar.success(f"✅ Using uploaded file: {uploaded_file.name}")
    else:
        st.sidebar.info("📂 Using default: Coronation Bakery Dataset.csv")
    
    # Initialize session state
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = []
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = ["Random Forest", "XGBoost"]
    if "selected_ts_models" not in st.session_state:
        st.session_state.selected_ts_models = ["Prophet", "ARIMA"]
    
    # Configuration sidebar
    with st.sidebar:
        st.header("Model Configuration")
        
        # Step 1: Select target variable
        target_variable = st.selectbox(
            "1. Select Target Variable:",
            options=get_valid_targets(df),
            index=get_valid_targets(df).index('Daily_Total') if 'Daily_Total' in get_valid_targets(df) else 0
        )
        
        # Step 2: Test size selection
        st.subheader("2. Test Set Size")
        test_size = st.slider(
            "Select test set size (%):",
            min_value=10,
            max_value=30,
            value=20,
            step=5
        ) / 100
        
        # Model type selection
        st.subheader("3. Model Type")
        model_type = st.radio(
            "Select model type:",
            ["Regression", "Time Series", "Both"],
            index=2
        )
        
        if model_type in ["Regression", "Both"]:
            st.subheader("4. Feature Selection")
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
            selected_features = st.multiselect(
                "Select features to include:",
                options=X.columns.tolist(),
                default=st.session_state.selected_features
            )
            
            # Update the session state with user's selection
            st.session_state.selected_features = selected_features
            
            # Regression model selection
            st.subheader("5. Select Regression Models")
            reg_model_options = [
                "Random Forest", "XGBoost", "Lasso", "Ridge", "ElasticNet",
                "SVR", "KNN", "Decision Tree", "AdaBoost", "Extra Trees",
                "MLP", "LightGBM", "CatBoost"
            ]
            st.session_state.selected_models = st.multiselect(
                "Choose regression models to run:",
                options=reg_model_options,
                default=st.session_state.selected_models
            )
        
        if model_type in ["Time Series", "Both"]:
            st.subheader("6. Select Time Series Models")
            ts_model_options = ["ARIMA", "SARIMA", "Exponential Smoothing", "Prophet"]
            st.session_state.selected_ts_models = st.multiselect(
                "Choose time series models to run:",
                options=ts_model_options,
                default=st.session_state.selected_ts_models
            )
        
        # Step 7: Run Analysis
        st.subheader("7. Run Analysis")
        run_button = st.button("Run Models", type="primary")
    
    # Data Overview Section
    st.header("📊 Data Overview")
    data_tabs = st.tabs(["Time Patterns", "Product Performance", "Seasonal Trends", "Autocorrelation"])
    
    with data_tabs[0]:
        st.subheader("Daily Sales Patterns")
        
        # Prepare time series plot
        if dates is not None and target_variable in original_df.columns:
            ts_df = pd.DataFrame({'Date': dates, target_variable: original_df[target_variable]})
            ts_df = ts_df.dropna()
            
            fig_ts = px.line(
                ts_df,
                x='Date',
                y=target_variable,
                title=f'{target_variable} Over Time',
                labels={target_variable: 'Value', 'Date': 'Date'}
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Add decomposition plot
            st.subheader("Time Series Decomposition")
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                decomposition = seasonal_decompose(
                    ts_df.set_index('Date')[target_variable],
                    model='additive',
                    period=7  # Weekly seasonality
                )
                
                fig_decomp = go.Figure()
                
                # Add observed
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.observed.index,
                        y=decomposition.observed,
                        name='Observed',
                        line=dict(color='blue')
                    )
                )
                
                # Add trend
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        name='Trend',
                        line=dict(color='red')
                    )
                )
                
                # Add seasonal
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal,
                        name='Seasonal',
                        line=dict(color='green')
                    )
                )
                
                # Add residual
                fig_decomp.add_trace(
                    go.Scatter(
                        x=decomposition.resid.index,
                        y=decomposition.resid,
                        name='Residual',
                        line=dict(color='purple')
                    )
                )
                
                fig_decomp.update_layout(
                    title='Time Series Decomposition',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend_title='Component',
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_decomp, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create decomposition plot: {str(e)}")
    
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
            
            # Product performance over time
            if dates is not None:
                product_ts_data = original_df.copy()
                product_ts_data['Date'] = dates
                product_ts_data = product_ts_data.groupby(['Date', 'Product_Type'])[target_variable].sum().reset_index()
                
                fig_product_ts = px.line(
                    product_ts_data,
                    x='Date',
                    y=target_variable,
                    color='Product_Type',
                    title=f'Product {target_variable} Over Time',
                    labels={target_variable: 'Value', 'Date': 'Date'}
                )
                st.plotly_chart(fig_product_ts, use_container_width=True)
    
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
            
            fig_month.update_layout(
                xaxis=dict(
                    type='category'
                )
            )
            st.plotly_chart(fig_month, use_container_width=True)
            
            # Add quarter analysis if we have enough data
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
    
    with data_tabs[3]:
        st.subheader("Autocorrelation Analysis")
        
        if target_variable in original_df.columns:
            try:
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                
                st.write("**Autocorrelation Function (ACF)**")
                fig_acf = plot_acf(original_df[target_variable].dropna(), lags=30)
                st.pyplot(fig_acf)
                
                st.write("**Partial Autocorrelation Function (PACF)**")
                fig_pacf = plot_pacf(original_df[target_variable].dropna(), lags=30)
                st.pyplot(fig_pacf)
            except Exception as e:
                st.warning(f"Could not create ACF/PACF plots: {str(e)}")
    
    # Model Analysis Section
    if run_button:
        if model_type in ["Regression", "Both"] and not st.session_state.selected_models:
            st.error("Please select at least one regression model to run")
        elif model_type in ["Time Series", "Both"] and not st.session_state.selected_ts_models:
            st.error("Please select at least one time series model to run")
        else:
            # Initialize results variables
            reg_results = None
            ts_results = None
            
            if model_type in ["Regression", "Both"]:
                with st.spinner("Preprocessing data for regression models..."):
                    df_processed = preprocess_data(df, target_variable)
                    X = df_processed[st.session_state.selected_features]
                    y = df_processed[target_variable]
                
                with st.spinner(f"Training {len(st.session_state.selected_models)} regression models..."):
                    reg_results, reg_models, X_test, y_test = train_regression_models(
                        X.values, y.values, int(len(X) * test_size), st.session_state.selected_models
                    )
                st.success("Regression model training completed!")
            
            if model_type in ["Time Series", "Both"]:
                with st.spinner(f"Training {len(st.session_state.selected_ts_models)} time series models..."):
                    ts_results, ts_models, ts_dates, ts_y = train_time_series_models(
                        dates, original_df[target_variable], test_size, st.session_state.selected_ts_models
                    )
                st.success("Time series model training completed!")
    
    # Display model results
    if run_button and (reg_results is not None or ts_results is not None):
        st.header("🔍 Model Analysis")
        
        if model_type in ["Regression", "Both"] and reg_results:
            st.subheader("Regression Models Performance")
            reg_metrics_df = pd.DataFrame.from_dict({
                name: {
                    'MSE': res['MSE'],
                    'R²': res['R2'],
                    'MAE': res['MAE']
                }
                for name, res in reg_results.items()
            }, orient='index')
            
            # Display metrics table
            st.dataframe(
                reg_metrics_df.style
                .background_gradient(cmap='Blues', subset=['R²'])
                .format({'MSE': '{:.2f}', 'R²': '{:.3f}', 'MAE': '{:.2f}'})
            )
            
            # Create comparison chart of R² scores
            fig_r2 = px.bar(
                reg_metrics_df.reset_index().rename(columns={'index': 'Model'}),
                x='Model',
                y='R²',
                title='Regression Model R² Score Comparison',
                color='R²',
                color_continuous_scale='Viridis',
                text='R²'
            )
            fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_r2, use_container_width=True)
            
            # Show actual vs predicted for best regression model
            if reg_results:
                best_reg_model = max(reg_results.items(), key=lambda x: x[1]['R2'])[0]
                st.subheader(f"Best Regression Model: {best_reg_model}")
                
                fig_reg_pred = go.Figure()
                fig_reg_pred.add_trace(go.Scatter(
                    y=y_test,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))
                fig_reg_pred.add_trace(go.Scatter(
                    y=reg_results[best_reg_model]['Predictions'],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', dash='dot')
                ))
                fig_reg_pred.update_layout(
                    title=f"{best_reg_model}: Actual vs Predicted",
                    xaxis_title="Observation",
                    yaxis_title=target_variable,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_reg_pred, use_container_width=True)
        
        if model_type in ["Time Series", "Both"] and ts_results:
            st.subheader("Time Series Models Performance")
            ts_metrics_df = pd.DataFrame.from_dict({
                name: {
                    'MSE': res['MSE'],
                    'R²': res['R2'],
                    'MAE': res['MAE']
                }
                for name, res in ts_results.items()
            }, orient='index')
            
            # Display metrics table
            st.dataframe(
                ts_metrics_df.style
                .background_gradient(cmap='Greens', subset=['R²'])
                .format({'MSE': '{:.2f}', 'R²': '{:.3f}', 'MAE': '{:.2f}'})
            )
            
            # Create comparison chart of R² scores
            fig_ts_r2 = px.bar(
                ts_metrics_df.reset_index().rename(columns={'index': 'Model'}),
                x='Model',
                y='R²',
                title='Time Series Model R² Score Comparison',
                color='R²',
                color_continuous_scale='Greens',
                text='R²'
            )
            fig_ts_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_ts_r2, use_container_width=True)
            
            # Show actual vs predicted for best time series model
            if ts_results:
                best_ts_model = max(ts_results.items(), key=lambda x: x[1]['R2'])[0]
                st.subheader(f"Best Time Series Model: {best_ts_model}")
                
                fig_ts_pred = go.Figure()
                fig_ts_pred.add_trace(go.Scatter(
                    x=ts_dates,
                    y=ts_y,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))
                fig_ts_pred.add_trace(go.Scatter(
                    x=ts_dates,
                    y=ts_results[best_ts_model]['Predictions'],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='green', dash='dot')
                ))
                fig_ts_pred.update_layout(
                    title=f"{best_ts_model}: Actual vs Predicted",
                    xaxis_title="Date",
                    yaxis_title=target_variable,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_ts_pred, use_container_width=True)
                
                # For Prophet, show components
                if best_ts_model == "Prophet":
                    try:
                        from prophet.plot import plot_components
                        fig_components = ts_models[best_ts_model].plot_components(ts_results[best_ts_model]['Forecast'])
                        st.pyplot(fig_components)
                    except:
                        pass
        
        # Compare best models from each category
        if model_type == "Both" and reg_results and ts_results:
            st.subheader("Best Model Comparison")
            
            best_reg = max(reg_results.items(), key=lambda x: x[1]['R2'])
            best_ts = max(ts_results.items(), key=lambda x: x[1]['R2'])
            
            comparison_df = pd.DataFrame({
                'Model Type': ['Regression', 'Time Series'],
                'Model': [best_reg[0], best_ts[0]],
                'R²': [best_reg[1]['R2'], best_ts[1]['R2']],
                'MSE': [best_reg[1]['MSE'], best_ts[1]['MSE']],
                'MAE': [best_reg[1]['MAE'], best_ts[1]['MAE']]
            })
            
            st.dataframe(
                comparison_df.style
                .background_gradient(cmap='Purples', subset=['R²'])
                .format({'R²': '{:.3f}', 'MSE': '{:.2f}', 'MAE': '{:.2f}'})
            )
            
            # Plot comparison
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=ts_dates,
                y=ts_y,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            fig_compare.add_trace(go.Scatter(
                x=ts_dates,
                y=reg_results[best_reg[0]]['Predictions'],
                mode='lines',
                name=f'Regression ({best_reg[0]})',
                line=dict(color='red', dash='dot')
            ))
            fig_compare.add_trace(go.Scatter(
                x=ts_dates,
                y=ts_results[best_ts[0]]['Predictions'],
                mode='lines',
                name=f'Time Series ({best_ts[0]})',
                line=dict(color='green', dash='dot')
            ))
            fig_compare.update_layout(
                title='Best Models Comparison',
                xaxis_title="Date",
                yaxis_title=target_variable,
                hovermode="x unified"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
    
    elif not run_button:
        st.info("Configure your models and click 'Run Models' to see results")

if __name__ == "__main__":
    main()
