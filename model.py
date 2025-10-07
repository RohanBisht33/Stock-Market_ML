# Author: RB  |  ML Pipeline for Financial Market Impact Prediction
# Predicting Stock_Impact_% using ML with feature engineering and time-based validation

# pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap joblib skl2onnx onnx onnxruntime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
np.random.seed(42)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data(filepath='stock_data.csv'):
    """Load CSV and display basic info."""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {filepath} | Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst 3 rows:\n{df.head(3)}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise

# =============================================================================
# 2. CLEAN DATA
# =============================================================================
def clean_data(df):
    """Remove duplicates, handle missing values, fix data types."""
    df = df.copy()
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"✓ Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  → Filled {col} with median")
            else:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                print(f"  → Filled {col} with mode/Unknown")
    
    # Convert Date to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"✓ Converted Date to datetime, sorted")
    
    # Ensure target column exists
    if 'Stock_Impact_%' not in df.columns:
        print("✗ Target column 'Stock_Impact_%' not found!")
        raise ValueError("Missing target column")
    
    print(f"✓ Cleaned data | Final shape: {df.shape}\n")
    return df

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def feature_engineering(df, target_col='Stock_Impact_%'):
    """Create time-based, ratio, lag, and rolling features."""
    df = df.copy()
    
    # Date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Ratio features
    df['Revenue_per_RD'] = df['AI_Revenue_USD_Mn'] / (df['R&D_Spending_USD_Mn'] + 1e-5)
    df['RD_to_Revenue_Ratio'] = df['R&D_Spending_USD_Mn'] / (df['AI_Revenue_USD_Mn'] + 1e-5)
    
    # Lag features (within company groups)
    for lag in [1, 3, 7]:
        df[f'AI_Revenue_Lag{lag}'] = df.groupby('Company')['AI_Revenue_USD_Mn'].shift(lag)
        df[f'RD_Spending_Lag{lag}'] = df.groupby('Company')['R&D_Spending_USD_Mn'].shift(lag)
        df[f'Stock_Impact_Lag{lag}'] = df.groupby('Company')[target_col].shift(lag)
    
    # Rolling features (7-day window)
    df['AI_Revenue_Roll7'] = df.groupby('Company')['AI_Revenue_USD_Mn'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['RD_Spending_Roll7'] = df.groupby('Company')['R&D_Spending_USD_Mn'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # Fill remaining NaNs from lag/rolling
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    print(f"✓ Feature engineering complete | New shape: {df.shape}")
    print(f"  → Added {df.shape[1] - 5} new features\n")
    return df

# =============================================================================
# 4. SPLIT & PREPROCESS
# =============================================================================
def prepare_data(df, target_col='Stock_Impact_%', test_size=0.2):
    """Time-based train/test split and preprocessing pipeline."""
    
    # Drop Date column (used for splits, not model)
    X = df.drop(columns=[target_col, 'Date'])
    y = df[target_col]
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Time-based split (last 20% as test)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")
    
    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough'
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features

# =============================================================================
# 5. TRAIN MODELS
# =============================================================================
def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train baseline and advanced models."""
    
    models = {}
    
    # Baseline: Linear Regression
    print("Training Baseline (Linear Regression)...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    models['Linear Regression'] = lr_pipeline
    y_pred_lr = lr_pipeline.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    print(f"  MAE: {mae_lr:.4f}\n")
    
    # RandomForest
    print("Training RandomForest Regressor...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_proc, y_train)
    models['RandomForest'] = (rf, preprocessor)
    y_pred_rf = rf.predict(X_test_proc)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    print(f"  MAE: {mae_rf:.4f}\n")
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train_proc, y_train)
    models['XGBoost'] = (xgb, preprocessor)
    y_pred_xgb = xgb.predict(X_test_proc)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    print(f"  MAE: {mae_xgb:.4f}\n")
    
    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=-1)
    lgb_model.fit(X_train_proc, y_train)
    models['LightGBM'] = (lgb_model, preprocessor)
    y_pred_lgb = lgb_model.predict(X_test_proc)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    print(f"  MAE: {mae_lgb:.4f}\n")
    
    return models, (X_test_proc, y_test), {'lr': y_pred_lr, 'rf': y_pred_rf, 'xgb': y_pred_xgb, 'lgb': y_pred_lgb}

# =============================================================================
# 6. EVALUATE MODELS
# =============================================================================
def evaluate_model(y_true, y_pred, model_name):
    """Calculate metrics: MAE, RMSE, R², MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'y_pred': y_pred}

# =============================================================================
# 7. PLOT RESIDUALS
# =============================================================================
def plot_residuals(y_true, y_pred_dict):
    """Plot residuals for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Model Residuals Analysis', fontsize=14, fontweight='bold')
    
    model_names = ['Linear Regression', 'RandomForest', 'XGBoost', 'LightGBM']
    y_preds = [y_pred_dict['lr'], y_pred_dict['rf'], y_pred_dict['xgb'], y_pred_dict['lgb']]
    
    for idx, (ax, name, y_pred) in enumerate(zip(axes.flat, model_names, y_preds)):
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residuals.png', dpi=100, bbox_inches='tight')
    print("\n✓ Residuals plot saved as 'residuals.png'")
    plt.close()

# =============================================================================
# 8. FEATURE IMPORTANCE
# =============================================================================
def plot_feature_importance(model, preprocessor, feature_names, model_name='Model'):
    """Plot feature importances for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print(f"\n→ {model_name} does not have feature_importances_")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'Feature {i}' for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 10 Features - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower()}.png', dpi=100, bbox_inches='tight')
    print(f"✓ Feature importance saved for {model_name}")
    plt.close()

# =============================================================================
# 9. SAVE MODEL
# =============================================================================
def save_model(model, model_name='final_model'):
    """Save model using joblib."""
    filepath = f'{model_name}.pkl'
    joblib.dump(model, filepath)
    print(f"\n✓ Model saved: {filepath}")
    return filepath

# =============================================================================
# 10. LOAD & PREDICT (Example)
# =============================================================================
def predict_single(model_path, sample_data, preprocessor=None):
    """Load model and make a single prediction."""
    model = joblib.load(model_path)
    
    # If model is a pipeline, preprocessor is included
    prediction = model.predict([sample_data])[0]
    print(f"\n✓ Prediction: {prediction:.4f}%")
    return prediction

# =============================================================================
# 11. EXPORT TO ONNX (Optional)
# =============================================================================
def export_to_onnx(model, X_sample, model_name='final_model'):
    """Convert trained model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        with open(f'{model_name}.onnx', 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"✓ ONNX model exported: {model_name}.onnx")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")

# =============================================================================
# 12. MAIN PIPELINE
# =============================================================================
def main():
    """Execute full ML pipeline."""
    print("\n" + "="*70)
    print("ML PIPELINE: Financial Market Impact Prediction")
    print("="*70 + "\n")
    
    # Load and clean
    df = load_data()
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor, num_feat, cat_feat = prepare_data(df)
    
    # Train models
    models, test_data, y_preds = train_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Evaluate
    results = {}
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    X_test_proc, y_test_proc = test_data
    results['Linear Regression'] = evaluate_model(y_test, y_preds['lr'], 'Linear Regression')
    results['RandomForest'] = evaluate_model(y_test_proc, y_preds['rf'], 'RandomForest')
    results['XGBoost'] = evaluate_model(y_test_proc, y_preds['xgb'], 'XGBoost')
    results['LightGBM'] = evaluate_model(y_test_proc, y_preds['lgb'], 'LightGBM')
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_r2 = results[best_model_name]['R2']
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")
    print(f"{'='*70}\n")
    
    # Visualizations
    plot_residuals(y_test_proc, y_preds)
    
    if best_model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
        if best_model_name == 'RandomForest':
            model_obj, _ = models[best_model_name]
        else:
            model_obj, _ = models[best_model_name]
        plot_feature_importance(model_obj, preprocessor, num_feat + cat_feat, best_model_name)
    
    # Save best model
    if best_model_name == 'Linear Regression':
        best_pipeline = models['Linear Regression']
        model_path = save_model(best_pipeline, 'best_model')
    else:
        best_pipeline = models[best_model_name]
        model_path = save_model(best_pipeline, 'best_model')
    
    # Export to ONNX
    print("\n→ ONNX Export Section (commented):")
    print("  # export_to_onnx(best_model, X_test_proc, 'best_model')")
    
    # Example prediction
    print(f"\n{'='*70}")
    print("EXAMPLE: Single Prediction")
    print(f"{'='*70}")
    sample = X_test_proc[0:1]
    print(f"Sample shape: {sample.shape}")
    if best_model_name == 'Linear Regression':
        pred = models['Linear Regression'].predict(X_test[0:1])[0]
    else:
        pred = models[best_model_name][0].predict(sample)[0]
    print(f"Sample prediction (Stock_Impact_%): {pred:.4f}")
    
    # FastAPI snippet (commented)
    print(f"\n{'='*70}")
    print("FASTAPI ENDPOINT (Snippet - Commented)")
    print(f"{'='*70}")
    print("""
# from fastapi import FastAPI
# import uvicorn
# 
# app = FastAPI()
# model = joblib.load('best_model.pkl')
# 
# @app.post("/predict")
# def predict(features: dict):
#     data = np.array([list(features.values())]).reshape(1, -1)
#     pred = model.predict(data)[0]
#     return {"prediction": pred}
# 
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
    """)
    
    # C++ ONNX Runtime snippet (commented)
    print(f"\n{'='*70}")
    print("C++ ONNX RUNTIME (Pseudo-Code)")
    print(f"{'='*70}")
    print("""
// #include <onnxruntime_cxx_api.h>
// 
// int main() {
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//     Ort::Session session(env, "best_model.onnx", Ort::SessionOptions{nullptr});
//     
//     // Prepare input
//     std::vector<float> input_data = {...}; // Your features
//     std::vector<int64_t> input_shape = {1, num_features};
//     
//     // Run inference
//     Ort::Value input = Ort::Value::CreateTensor<float>(...);
//     auto output = session.Run(...);
//     
//     float result = output[0].GetTensorMutableData<float>()[0];
//     std::cout << "Prediction: " << result << std::endl;
//     return 0;
// }
    """)
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

# Done — RB