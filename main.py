# main.py
from src import config
from src.data_loader import load_data, expand_sales, tag_promotions, cluster_by_avg, assign_clusters
from src.feature_engineering import engineer_features
from src.visualizer import visualize_all
from src.forecaster import forecast_promotion5, evaluate_forecast, summarize_clusters, export_forecast
from src.model import train_model
from src.visualizer import visualize_all

def main():
    # Load + prep
    sales_a, sales_b, promos, product_groups = load_data()
    promo5 = promos[promos["Period"] == "Promo5"].iloc[0]
    sales_full = expand_sales(sales_a, expand=config.EXPAND_SALES)
    sales_full, sales_b, sales_b_promo5 = tag_promotions(sales_full, sales_b, promos)

    # Clustering
    item_clusters = cluster_by_avg(sales_full, "Item", "Item")
    store_clusters = cluster_by_avg(sales_full, "Store", "Store")
    sales_full, sales_b = assign_clusters(sales_full, sales_b, item_clusters, store_clusters)

    # Feature engineering
    sales_full, sales_b = engineer_features(sales_full, sales_b, enable=config.FEATURE_ENGINEERING)

    visualize_all(sales_full, sales_b, promos.head(4), promos[promos["Period"] == "Promo5"].iloc[0],
                  product_groups, item_clusters, store_clusters, config.FIGS_DIR, config.FEATURE_ENGINEERING)
    
    # Compute cluster lift (used by baseline forecaster)
    item_cluster_lift = sales_full.groupby(["ItemCluster", "Promotion"])["Quantity"] \
        .mean().unstack().assign(Lift=lambda x: x[True] - x[False])
    
    # Forecast and evaluate
    sales_b, sales_b_promo5 = forecast_promotion5(sales_full, sales_b, promos, item_cluster_lift, item_clusters, store_clusters)
    _ = evaluate_forecast(sales_b, sales_b_promo5)
    summarize_clusters(sales_full, "Train (Promo1-4 period)")
    summarize_clusters(sales_b_promo5, "Test (Promo5 period)")
    export_forecast(sales_b_promo5)

    model, sales_b, sales_b_promo5 = train_model(
        sales_full, sales_b, sales_b_promo5,
        config.FEATURES, config.RESULTS_DIR, config.MODELS_DIR, config.FIGS_DIR,
        promos.head(4), promos[promos["Period"] == "Promo5"].iloc[0],
        model_name=config.MODEL_NAME,
    )
    # Visual diagnostics
    visualize_all(sales_full, sales_b, promos.head(4), promo5, product_groups,
                  item_clusters, store_clusters, config.FIGS_DIR, config.FEATURE_ENGINEERING)

if __name__ == "__main__":
    main()