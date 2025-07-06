# src/config.py
import os
import matplotlib as mpl
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Directory paths
FIGS_DIR = "figures"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Plotting config
mpl.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 20
})

# Plot color palette
COLOR_ORANGE = '#d95f02'
COLOR_BLUE = '#1b5e9e'
COLOR_LBLUE = '#a6cee3'
COLOR_GOLD = 'goldenrod'

# Global flags
EXPAND_SALES = False
FEATURE_ENGINEERING = True

# Feature list for modeling
FEATURES = [
    'Date', 'Store', 'Item', 'Promotion', 'ItemCluster', 'Quantity',
    'StoreCluster', 'DayOfWeek', 'Last7Avg', 'Last30Avg', 'LastSaleDayDiff',
    'PromoStartLag', 'isWeekend'
]

CATEGORICAL_FEATURES = ["ItemCluster", "StoreCluster"]

MODEL_NAME = 'LGBM'
SAMPLE_SIZE = -1

# Save format (optional utility)
MODEL_SAVE_FORMATS = {
    'LGBM': 'txt',
    'XGB': 'json',
    'CATBOOST': 'cbm'
}
