import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def train_model(
    sales_full, sales_b, sales_b_promo5,
    FEATURES, results_dir, models_dir, figs_dir,
    promos1to4, promo5,
    model_name='LGBM'
):
    print("\n" + "="*50)
    print("   🤖 Training and Evaluating Model...   ")
    print("="*50 + "\n")

    # Prepare train/val/test
    X = sales_full[FEATURES].drop(columns=["Quantity", "Date"])
    y = sales_full["Quantity"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    X_test = sales_b[FEATURES].drop(columns=["Quantity", "Date"])
    y_test = sales_b["Quantity"]
    X_test_promo5 = sales_b_promo5[FEATURES].drop(columns=["Quantity", "Date"])
    y_test_promo5 = sales_b_promo5["Quantity"]

    # Cast categoricals and handle missing clusters
    cat_features = ["ItemCluster", "StoreCluster"]
    for col in cat_features:
        X_train = X_train[X_train[col].notnull()]
        y_train = y_train.loc[X_train.index]
        X_val = X_val[X_val[col].notnull()]
        y_val = y_val.loc[X_val.index]
        X_test[col] = X_test[col].fillna("Unknown")
        X_test_promo5[col] = X_test_promo5[col].fillna("Unknown")

    for df in [X_train, X_val, X_test, X_test_promo5]:
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category")

    # Train model
    if model_name == "LGBM":
        model = lgb.LGBMRegressor(
            objective="regression", n_estimators=100, learning_rate=0.1,
            early_stopping_rounds=10, verbose=0, num_leaves=31,
            num_iterations=100
        )
    elif model_name == "XGB":
        model = XGBRegressor(
            objective="reg:squarederror", n_estimators=100,
            enable_categorical=True, eval_metric="rmse", verbosity=0
        )
    elif model_name == "CATBOOST":
        model = CatBoostRegressor(
            iterations=100, learning_rate=0.1, depth=6,
            loss_function='RMSE', verbose=0, early_stopping_rounds=10,
            cat_features=cat_features
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    if not X_test_promo5.empty:
        y_test_promo5_pred = model.predict(X_test_promo5)
    else:
        print("\n" + "!"*60)
        print("   ⚠️  Warning: Promo5 test set is empty. Skipping evaluation for Promo5.")
        print("!"*60 + "\n")
        y_test_promo5_pred = np.full(len(y_test_promo5), np.nan)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_nrmse = test_rmse / (y_test.max() - y_test.min())

    test_promo5_mae = mean_absolute_error(y_test_promo5, y_test_promo5_pred)
    test_promo5_rmse = root_mean_squared_error(y_test_promo5, y_test_promo5_pred)
    test_promo5_nrmse = test_promo5_rmse / (y_test_promo5.max() - y_test_promo5.min())

    print("\n" + "="*50)
    print(f"   📈 {model_name} Model Validation Results:")
    print("="*50)
    print(f"   Validation MAE:   {val_mae:.4f}")
    print(f"   Validation RMSE:  {val_rmse:.4f}")
    print("\n" + "="*50)
    print(f"   📈 {model_name} Model Test Results:")
    print("="*50)
    print(f"   Test MAE:         {test_mae:.4f}")
    print(f"   Test RMSE:        {test_rmse:.4f}")
    print(f"   Test NRMSE:       {test_nrmse:.4f}")
    print("\n" + "="*50)
    print(f"   📈 {model_name} Model Promo5 Results:")
    print("="*50)
    print(f"   Test Promo5 MAE:  {test_promo5_mae:.4f}")
    print(f"   Test Promo5 RMSE: {test_promo5_rmse:.4f}")
    print(f"   Test Promo5 NRMSE:{test_promo5_nrmse:.4f}")

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    if model_name == 'LGBM':
        model.booster_.save_model(os.path.join(models_dir, f"{model_name}.txt"))
    elif model_name == 'XGB':
        model.save_model(os.path.join(models_dir, f"{model_name}.json"))

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    # Write model evaluation results to a file
    results_file = os.path.join(results_dir, f"{model_name}_results.txt")
    with open(results_file, "w") as f:
        f.write("="*50 + "\n")
        f.write(f"📈 {model_name} Model Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Validation MAE:   {val_mae:.4f}\n")
        f.write(f"Validation RMSE:  {val_rmse:.4f}\n")
        f.write("\n")
        f.write(f"Test MAE:         {test_mae:.4f}\n")
        f.write(f"Test RMSE:        {test_rmse:.4f}\n")
        f.write(f"Test NRMSE:       {test_nrmse:.4f}\n")
        f.write("\n")
        f.write(f"Promo5 MAE:       {test_promo5_mae:.4f}\n")
        f.write(f"Promo5 RMSE:      {test_promo5_rmse:.4f}\n")
        f.write(f"Promo5 NRMSE:     {test_promo5_nrmse:.4f}\n")
        f.write("="*50 + "\n")

    # Save predictions for future visualization
    sales_b["PredictedQuantity"] = y_test_pred
    sales_b_promo5["PredictedQuantity"] = y_test_promo5_pred
    sales_b.to_csv(os.path.join(results_dir, f"{model_name}_test_predictions.csv"), index=False)
    sales_b_promo5.to_csv(os.path.join(results_dir, f"{model_name}_promo5_predictions.csv"), index=False)

    return model, sales_b, sales_b_promo5