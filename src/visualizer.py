import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lightgbm as lgb

def visualize_all(sales_full, sales_b, promos1to4, promo5, product_groups, item_clusters, store_clusters, figs_dir, FEATURE_ENGINEERING):
    print("\n" + "="*40)
    print("   📊 Visualizing Data...   ")
    print("="*40 + "\n")
    os.makedirs(figs_dir, exist_ok=True)
    
    # Plot 1: Total Sales Over Time
    def plot_sales_over_time():
        plt.figure(figsize=(14, 6))
        plt.plot(sales_full.groupby("Date")["Quantity"].sum(), label="Train", color="blue")
        plt.plot(sales_b.groupby("Date")["Quantity"].sum(), label="Test", color="orange")
        for _, promo in promos1to4.iterrows():
            plt.axvspan(promo["StartDate"], promo["EndDate"], color='blue', alpha=0.2)
        plt.axvspan(promo5["StartDate"], promo5["EndDate"], color='red', alpha=0.2)
        plt.title("Sales Quantity Over Time")
        plt.xlabel("Date"); plt.ylabel("Total Quantity")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "sales_quantity_over_time.png"))

    # Plot 2: Cluster Distributions
    def plot_item_cluster_dist():
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        sns.countplot(data=sales_full, x="ItemCluster", order=["Slow", "Medium", "Fast"], ax=axs[0])
        axs[0].set_title("Train"); sns.countplot(data=sales_b, x="ItemCluster", order=["Slow", "Medium", "Fast"], ax=axs[1])
        axs[1].set_title("Test"); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "item_clusters_distribution.png"))

    # Plot 3: Return Rate
    def plot_return_rate():
        if not FEATURE_ENGINEERING:
            return
        sales_full["TotalReturn"] = sales_full["Quantity"].apply(lambda x: -x if x < 0 else 0)
        total_return = sales_full.groupby("PromoStartLag")["TotalReturn"].sum().iloc[:50]
        plt.figure(figsize=(12, 6)); total_return.plot(kind='bar', color='#d95f02')
        plt.yscale("log"); plt.xlabel("Promo Start Lag (days)"); plt.ylabel("Total Return (log)")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir, "total_return_rate.png"))

    # Cluster Visualization
    def plot_cluster_scatter(df, group, cluster_map):
        avg = df[df["Promotion"] == False].groupby(group)["Quantity"].mean().reset_index()
        avg["Cluster"] = avg[group].map(cluster_map)
        q33, q66 = avg["Quantity"].quantile([0.33, 0.66])
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=range(len(avg)), y="Quantity", data=avg, hue="Cluster", palette="Set1")
        plt.axhline(y=q33, color="gray", linestyle="--")
        plt.axhline(y=q66, color="black", linestyle="--")
        plt.yscale("log"); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"{group.lower()}_cluster_scatter.png"))

    # Lift Visuals
    def compute_and_plot_lift(col):
        avg = sales_full[sales_full["Promotion"] == False].groupby(col)["Quantity"].mean().reset_index(name="NonPromoAvg")
        avg_promo = sales_full[sales_full["Promotion"] == True].groupby(col)["Quantity"].mean().reset_index(name="PromoAvg")
        merged = pd.merge(avg, avg_promo, on=col, how="inner")
        merged["PromotionLift"] = merged["PromoAvg"] - merged["NonPromoAvg"]
        top = merged.sort_values("PromotionLift", ascending=False).head(20)
        plt.figure(figsize=(10, 10))
        sns.barplot(data=top, x=col, y="PromotionLift", palette="viridis")
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"top_{col.lower()}_lift.png"))

    # Cluster Lift Plots
    def plot_lift_by_cluster(cluster_type, cluster_lift):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cluster_lift.index, y="Lift", data=cluster_lift.reset_index(), palette="viridis")
        plt.xlabel(f"{cluster_type} Cluster"); plt.ylabel("Avg Promotion Lift")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir, f"lift_{cluster_type.lower()}_cluster.png"))

    # Feature Importance
    def plot_feature_importance(model, model_name):
        if model_name == "LGBM":
            lgb.plot_importance(model, max_num_features=30, grid=False, figsize=(12, 6))
            plt.tight_layout(); plt.savefig(os.path.join(figs_dir, f"{model_name}_feature_importance.png"))

    # Execute
    plot_sales_over_time()
    plot_item_cluster_dist()
    plot_return_rate()
    plot_cluster_scatter(sales_full, "Item", item_clusters)
    plot_cluster_scatter(sales_full, "Store", store_clusters)
    compute_and_plot_lift("Item")
    compute_and_plot_lift("Store")

    item_cluster_lift = sales_full.groupby(["ItemCluster", "Promotion"])["Quantity"].mean().unstack().assign(Lift=lambda x: x[True] - x[False])
    store_cluster_lift = sales_full.groupby(["StoreCluster", "Promotion"])["Quantity"].mean().unstack().assign(Lift=lambda x: x[True] - x[False])
    plot_lift_by_cluster("Item", item_cluster_lift)
    plot_lift_by_cluster("Store", store_cluster_lift)

    return item_cluster_lift
