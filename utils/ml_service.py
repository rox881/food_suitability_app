import pandas as pd
import joblib
import os
import numpy as np

# ----------------------------
# Loading Model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_food_suitability_model_v1.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ----------------------------
# Logic
# ----------------------------
def predict_suitability(user, food, food_source):
    """
    Combines ML Model prediction with hard rules (Hybrid Logic).
    """
    
    # 1. Prepare ML Input
    X = pd.DataFrame([{
        "Weight": user["weight"],
        "BMI": user["bmi"],
        "Calories": food["calories"],
        "Protein": food["protein"],
        "Fat": food["fat"],
        "Carbohydrates": food["carbs"],
    }])
    
    # 2. Get ML Prediction
    if model:
        pred = int(model.predict(X)[0])
        conf = float(model.predict_proba(X)[0][pred])
        ml_result = "Suitable" if pred == 1 else "Not Suitable"
    else:
        # Fallback if model missing
        ml_result = "Unknown"
        conf = 0.0

    # 3. Hybrid Logic (Rule Engine)
    # Why? The model ignores Weight/BMI (based on importance analysis).
    # We add safety rules to override the model.
    final_result = ml_result
    final_conf = round(conf, 3)
    override_reason = None
    
    # Rule A: High BMI + High Calorie/Fat Food -> Risk
    if user["bmi"] > 30:
        if food["fat"] > 15 or food["calories"] > 400:
            final_result = "Not Suitable"
            final_conf = 0.95
            override_reason = "High Calorie/Fat food is risky for BMI > 30"

    # Rule B: Underweight + Low Calorie -> Suggest High Calorie
    if user["bmi"] < 18.5:
        if food["calories"] < 100:
            # Not strict "Not Suitable", but maybe a warning?
            # For now, we trust the model unless it's obviously bad.
            pass

    # Rule C: Source-based overrides (Legacy logic)
    if food_source in ["fried", "fast_food", "sweet"]:
        final_result = "Not Suitable"
        final_conf = 1.0
        override_reason = f"Category '{food_source}' is generally unhealthy"
    
    elif food_source in ["fruit", "vegetable"]:
        final_result = "Suitable" 
        final_conf = 1.0
        override_reason = f"Category '{food_source}' is generally healthy"

    return final_result, final_conf, override_reason

