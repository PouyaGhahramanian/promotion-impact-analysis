# src/forecaster.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def forecast_promotion5(sales_full, sales_b, promos, item_cluster_lift, item_clusters, store_clusters):
    print("\n" + "="*40)
    print("   📈 Forecasting Promotion 5...   ")
    print("="*40 + "\n")

    # Assign clusters
    sales_b["ItemCluster"] = sales_b["Item"].map(item_clusters)
    sales_b["StoreCluster"] = sales_b["Store"].map(store_clusters)

    # Identify Promotion 5 period
    promo5 = promos[promos["Period"] == "Promo5"].iloc[0]
    sales_b["Promotion5"] = (sales_b["Date"] >= promo5["StartDate"]) & (sales_b["Date"] <= promo5["EndDate"])

    # Compute baseline and lift
    baseline_by_item_cluster = sales_full[sales_full["Promotion"] == False].groupby("ItemCluster")["Quantity"].mean()
    if "ItemCluster" not in item_cluster_lift.columns:
        item_cluster_lift = item_cluster_lift.reset_index()
    lift_by_item_cluster = item_cluster_lift.set_index("ItemCluster")["Lift"]

    # Warn if any cluster is missing
    missing_clusters = set(sales_b["ItemCluster"].unique()) - set(lift_by_item_cluster.index)
    if missing_clusters:
        print(f"⚠️ Warning: Missing lift values for clusters: {missing_clusters}. Defaulting lift to 0.")

    # Forecast
    sales_b["ExpectedQuantity"] = (
        sales_b["ItemCluster"].map(baseline_by_item_cluster)
        + sales_b["ItemCluster"].map(lift_by_item_cluster).fillna(0)
    )
    sales_b_promo5 = sales_b[sales_b["Promotion5"]].copy()

    return sales_b, sales_b_promo5

def evaluate_forecast(sales_b, sales_b_promo5):
    print("\n" + "="*50)
    print("   📊 Evaluating Forecast for Promotion 5...   ")
    print("="*50 + "\n")
    y_true = sales_b_promo5["Quantity"]
    y_pred = sales_b_promo5["ExpectedQuantity"]
    y_true_all = sales_b["Quantity"]
    y_pred_all = sales_b["ExpectedQuantity"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae_all = mean_absolute_error(y_true_all, y_pred_all)
    rmse_all = root_mean_squared_error(y_true_all, y_pred_all)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    nrmse = rmse / (y_true.max() - y_true.min())

    print("\n" + "="*50)
    print("   📊 Forecast Evaluation on Promotion 5:")
    print("="*50)
    print(f"   MAE (promo5)     = {mae:.4f}")
    print(f"   RMSE (promo5)    = {rmse:.4f}")
    print(f"   MAE (all test)   = {mae_all:.4f}")
    print(f"   RMSE (all test)  = {rmse_all:.4f}")
    print(f"   MAPE             = {mape:.2f}%")
    print(f"   NRMSE            = {nrmse:.2f}%")
    print("="*50)

    print("\n" + "-"*50)
    print("   📏 Reference Scale:")
    print("-"*50)
    print(f"   Avg y_true   = {y_true.mean():.4f}")
    print(f"   Std y_true   = {y_true.std():.4f}")
    print(f"   Max y_true   = {y_true.max():.4f}")
    print(f"   Min y_true   = {y_true.min():.4f}")
    print("-"*50)

    return {
        "mae": mae, "rmse": rmse, "mae_all": mae_all, "rmse_all": rmse_all, "mape": mape, "nrmse": nrmse
    }

def summarize_clusters(df, dataset_name="Train"):
    print("\n" + "="*50)
    print(f"   📊 {dataset_name} Item Cluster Summary:")
    print("="*50)
    print(df.groupby("ItemCluster")["Quantity"].agg(["count", "mean", "std"]))

    print("\n" + "="*50)
    print(f"   📊 {dataset_name} Store Cluster Summary:")
    print("="*50)
    print(df.groupby("StoreCluster")["Quantity"].agg(["count", "mean", "std"]))

def export_forecast(sales_b_promo5, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    sales_b_promo5.to_csv(os.path.join(results_dir, "promotion5_forecast_naive.csv"), index=False)