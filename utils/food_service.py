import pandas as pd
import difflib
import os

# ----------------------------
# Loading Data
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_B_cleaned.csv")

try:
    food_df = pd.read_csv(DATA_PATH)
    food_df["Food"] = food_df["Food"].astype(str).str.strip().str.lower()
except Exception as e:
    print(f"Error loading dataset: {e}")
    food_df = pd.DataFrame()

# ----------------------------
# Constants
# ----------------------------
CATEGORY_FALLBACK = {
    "fruit": {"calories": 80, "protein": 1, "fat": 0.3, "carbs": 20},
    "vegetable": {"calories": 50, "protein": 2, "fat": 0.2, "carbs": 10},
    "grain": {"calories": 180, "protein": 5, "fat": 1, "carbs": 38},
    "dairy": {"calories": 150, "protein": 8, "fat": 8, "carbs": 12},
    "meat": {"calories": 250, "protein": 22, "fat": 18, "carbs": 0},
    "fried": {"calories": 350, "protein": 18, "fat": 22, "carbs": 20},
    "fast_food": {"calories": 320, "protein": 14, "fat": 18, "carbs": 30},
    "sweet": {"calories": 400, "protein": 4, "fat": 20, "carbs": 50},
    "beverage": {"calories": 120, "protein": 1, "fat": 0, "carbs": 28},
}

CATEGORY_KEYWORDS = {
    "fruit": ["apple", "banana", "orange", "mango"],
    "vegetable": ["spinach", "carrot", "broccoli"],
    "grain": ["rice", "bread", "roti", "chapati", "oats", "pasta"],
    "dairy": ["milk", "cheese", "curd", "paneer", "butter"],
    "meat": ["chicken", "meat", "beef", "fish", "egg"],
    "fried": ["fried", "pakora", "samosa"],
    "fast_food": ["pizza", "burger", "fries"],
    "sweet": ["cake", "sweet", "chocolate", "ice cream"],
    "beverage": ["juice", "cola", "soda", "drink"],
}

HIGH_RISK_CATEGORIES = ["fried", "fast_food", "sweet"]
ALWAYS_SAFE_CATEGORIES = ["fruit", "vegetable"]

# ----------------------------
# Logic
# ----------------------------
def get_food_nutrition(food_name):
    food_name = food_name.strip().lower()

    if not food_name:
        return None, None, None

    # 1️⃣ Exact dataset match
    exact = food_df[food_df["Food"] == food_name]
    if not exact.empty:
        return _extract_nutrition(exact.iloc[0]), "dataset", food_name

    # 2️⃣ Category match (Prioritized over fuzzy)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k == food_name for k in keywords):
            return CATEGORY_FALLBACK[category], category, category
        
    # 3️⃣ Fuzzy match
    possible_matches = difflib.get_close_matches(food_name, food_df["Food"].tolist(), n=1, cutoff=0.6)
    if possible_matches:
        match_name = possible_matches[0]
        row = food_df[food_df["Food"] == match_name].iloc[0]
        return _extract_nutrition(row), "dataset (fuzzy)", match_name

    # 4️⃣ Loose Category fallback
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in food_name for k in keywords):
            return CATEGORY_FALLBACK[category], category, category

    return None, None, None

def _extract_nutrition(row):
    return {
        "calories": float(row["Calories"]),
        "protein": float(row["Protein"]),
        "fat": float(row["Fat"]),
        "carbs": float(row["Carbohydrates"]),
    }
