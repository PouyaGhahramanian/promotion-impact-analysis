# src/data_loader.py
import pandas as pd

def load_data():
    print("\n" + "="*30)
    print("       Loading Data...       ")
    print("="*30 + "\n")
    sales_a = pd.read_csv("data/assignment4.1a.csv", parse_dates=["Date"])
    sales_b = pd.read_csv("data/assignment4.1b.csv", parse_dates=["Date"])
    promos1to4 = pd.read_csv("data/PromotionDates.csv", parse_dates=["StartDate", "EndDate"], nrows=4)
    promos5to6 = pd.read_csv("data/PromotionDates.csv", skiprows=range(1, 5), header=0,
                             parse_dates=["StartDate", "EndDate"], dayfirst=True)
    promos = pd.concat([promos1to4, promos5to6], ignore_index=True)

    # Add promo period names (assumed based on assignment)
    promos["Period"] = ["Promo1", "Promo2", "Promo3", "Promo4", "Promo5", "Promo6"]

    sales_a.columns = sales_b.columns = ["Date", "Store", "Item", "Quantity"]
    product_groups = pd.read_csv("data/assignment4.1c.csv")
    return sales_a, sales_b, promos, product_groups

def expand_sales(sales_a, expand=False):
    if not expand:
        return sales_a.copy()
    print("\n" + "-"*40)
    print("   📂 Expanding Sales Data...   ")
    print("-"*40 + "\n")
    observed_pairs = sales_a[["Store", "Item"]].drop_duplicates()
    all_dates = pd.date_range(sales_a["Date"].min(), sales_a["Date"].max())
    full_index = pd.MultiIndex.from_frame(
        observed_pairs.assign(key=1).merge(pd.DataFrame({"Date": all_dates, "key": 1}), on="key").drop(columns="key")
    )
    return sales_a.groupby(["Store", "Item", "Date"])["Quantity"].sum().reindex(full_index, fill_value=0).reset_index()

def tag_promotions(sales_full, sales_b, promos):
    print("\n" + "-"*40)
    print("   🚀 Tagging Promotions in Sales Data...   ")
    print("-"*40 + "\n")
    sales_full["Promotion"] = False
    for _, row in promos.head(4).iterrows():
        sales_full.loc[(sales_full["Date"] >= row["StartDate"]) & (sales_full["Date"] <= row["EndDate"]), "Promotion"] = True

    promo5 = promos[promos["Period"] == "Promo5"].iloc[0]
    sales_b["Promotion5"] = (sales_b["Date"] >= promo5["StartDate"]) & (sales_b["Date"] <= promo5["EndDate"])
    sales_b_promo5 = sales_b[sales_b["Promotion5"]].copy()

    sales_b["Promotion"] = False
    for _, row in promos.tail(2).iterrows():
        sales_b.loc[(sales_b["Date"] >= row["StartDate"]) & (sales_b["Date"] <= row["EndDate"]), "Promotion"] = True

    return sales_full, sales_b, sales_b_promo5

def cluster_by_avg(df, group_col, label):
    print("\n" + "*"*50)
    print(f"   📊 Clustering {label.lower()}s by Average Sales Quantity...   ")
    print("*"*50 + "\n")
    avg = df[df["Promotion"] == False].groupby(group_col)["Quantity"].mean()
    q33, q66 = avg.quantile([0.33, 0.66])
    cluster = avg.apply(lambda x: "Slow" if x <= q33 else "Medium" if x <= q66 else "Fast")
    print("\n" + "="*50)
    print(f"   📈 {label} Thresholds:")
    print(f"   Slow ≤ {q33:.4f}")
    print(f"   Medium ≤ {q66:.4f}")
    print(f"   Fast > {q66:.4f}")
    print("="*50 + "\n")
    return cluster

def assign_clusters(sales_full, sales_b, item_clusters, store_clusters):
    sales_full = sales_full.copy()
    sales_b = sales_b.copy()
    sales_full["ItemCluster"] = sales_full["Item"].map(item_clusters)
    sales_full["StoreCluster"] = sales_full["Store"].map(store_clusters)
    sales_b["ItemCluster"] = sales_b["Item"].map(item_clusters)
    sales_b["StoreCluster"] = sales_b["Store"].map(store_clusters)
    return sales_full, sales_b