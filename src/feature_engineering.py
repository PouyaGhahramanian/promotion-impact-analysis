# src/feature_engineering.py
import os
import pandas as pd

DATA_DIR = "data"

def engineer_features(sales_full, sales_b, enable=True):
    if not enable:
        print("\n" + "~"*50)
        print("   ⚙️  Skipping Feature Engineering as per Settings   ")
        print("~"*50 + "\n")
        return sales_full, sales_b

    print("\n" + "="*50)
    print("   ⚙️  Performing Feature Engineering...   ")
    print("="*50 + "\n")
    full_path = os.path.join(DATA_DIR, "sales_full_enhanced.csv")
    b_path = os.path.join(DATA_DIR, "sales_b_enhanced.csv")

    if os.path.exists(full_path) and os.path.exists(b_path):
        print("\n" + "-"*60)
        print("   📂 Enhanced Sales Data Already Exists. Loading from CSV Files...   ")
        print("-"*60 + "\n")
        sales_full = pd.read_csv(full_path, parse_dates=["Date"])
        sales_b = pd.read_csv(b_path, parse_dates=["Date"])

        # ensure proper sorting even when loading
        sales_full.sort_values(by=["Store", "Item", "Date"], inplace=True)
        sales_b.sort_values(by=["Store", "Item", "Date"], inplace=True)

        return sales_full, sales_b

    # Day of week
    sales_full["DayOfWeek"] = sales_full["Date"].dt.dayofweek
    sales_b["DayOfWeek"] = sales_b["Date"].dt.dayofweek

    # Rolling averages
    sales_full["Last7Avg"] = sales_full.groupby(["Store", "Item"])["Quantity"].transform(lambda x: x.rolling(7, 1).mean())
    sales_b["Last7Avg"] = sales_b.groupby(["Store", "Item"])["Quantity"].transform(lambda x: x.rolling(7, 1).mean())
    sales_full["Last30Avg"] = sales_full.groupby(["Store", "Item"])["Quantity"].transform(lambda x: x.rolling(30, 1).mean())
    sales_b["Last30Avg"] = sales_b.groupby(["Store", "Item"])["Quantity"].transform(lambda x: x.rolling(30, 1).mean())

    # Days since last sale
    sales_full["LastSaleDayDiff"] = sales_full.groupby(["Store", "Item"])["Date"].transform(lambda x: (x - x.shift()).dt.days.fillna(0))
    sales_b["LastSaleDayDiff"] = sales_b.groupby(["Store", "Item"])["Date"].transform(lambda x: (x - x.shift()).dt.days.fillna(0))

    # Sort by Store-Item-Date for reliable transformations
    sales_full.sort_values(by=["Store", "Item", "Date"], inplace=True)
    sales_b.sort_values(by=["Store", "Item", "Date"], inplace=True)

    # PromoStart-related features
    for df in [sales_full, sales_b]:
        if "Promotion" not in df.columns:
            raise ValueError("Expected 'Promotion' column to be present. Did you forget to call tag_promotions()?")

        df["PromoStart"] = df.groupby(["Store", "Item"])["Promotion"].transform(lambda x: (x == 1) & (x.shift().fillna(0) != 1))
        df["LastPromoStartDate"] = df["Date"].where(df["PromoStart"])
        df["LastPromoStartDate"] = df.groupby(["Store", "Item"])["LastPromoStartDate"].ffill()
        df["PromoStartLag"] = (df["Date"] - df["LastPromoStartDate"]).dt.days.fillna(0)
        df["isWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Save enhanced data
    sales_full.to_csv(full_path, index=False)
    sales_b.to_csv(b_path, index=False)

    return sales_full, sales_b