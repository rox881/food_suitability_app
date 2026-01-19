from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# ----------------------------
# App initialization
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "random_forest_food_suitability_model_v1.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "dataset_B_cleaned.csv"
)

# ----------------------------
# Load model & dataset
# ----------------------------
model = joblib.load(MODEL_PATH)

food_df = pd.read_csv(DATA_PATH)
food_df["Food"] = food_df["Food"].astype(str).str.strip().str.lower()

# ----------------------------
# Category definitions
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
# User input validation
# ----------------------------
def get_user_input(form):
    try:
        weight = float(form.get("weight", ""))
        bmi = float(form.get("bmi", ""))
    except ValueError:
        return None, "Weight and BMI must be numeric values."

    if not (30 <= weight <= 200):
        return None, "Weight must be between 30 and 200 kg."

    if not (12 <= bmi <= 45):
        return None, "BMI must be between 12 and 45."

    return {"weight": weight, "bmi": bmi}, None

# ----------------------------
# Food resolution logic
# ----------------------------
def get_food_nutrition(food_name):
    food_name = food_name.strip().lower()

    if not food_name:
        return None, None

    # 1️⃣ Exact dataset match
    exact = food_df[food_df["Food"] == food_name]
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "calories": float(row["Calories"]),
            "protein": float(row["Protein"]),
            "fat": float(row["Fat"]),
            "carbs": float(row["Carbohydrates"]),
        }, "dataset"

    # 2️⃣ Category fallback
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in food_name for k in keywords):
            return CATEGORY_FALLBACK[category], category

    return None, None

# ----------------------------
# Create model input
# ----------------------------
def create_model_input(user, food):
    return pd.DataFrame([{
        "Weight": user["weight"],
        "BMI": user["bmi"],
        "Calories": food["calories"],
        "Protein": food["protein"],
        "Fat": food["fat"],
        "Carbohydrates": food["carbs"],
    }])

# ----------------------------
# Route
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = error = None

    if request.method == "POST":
        try:
            # Step 1: Validate user
            user, error = get_user_input(request.form)
            if error:
                return render_template("index.html", error=error)

            # Step 2: Resolve food
            food_name = request.form.get("food", "")
            food, food_source = get_food_nutrition(food_name)

            if food is None:
                return render_template(
                    "index.html",
                    error="Food not supported. Try common food items."
                )

            # Step 3: ML prediction
            X = create_model_input(user, food)
            pred = int(model.predict(X)[0])
            conf = float(model.predict_proba(X)[0][pred])

            ml_result = "Suitable" if pred == 1 else "Not Suitable"
            confidence = round(conf, 3)

            # ----------------------------
            # FINAL DECISION (STABLE v1)
            # ----------------------------
            if food_source in HIGH_RISK_CATEGORIES:
                result = "Not Suitable"
                confidence = 1.0

            elif food_source in ALWAYS_SAFE_CATEGORIES:
                result = "Suitable"
                confidence = 1.0

            else:
                result = ml_result

            # Debug logs
            print("USER:", user)
            print("FOOD:", food_name)
            print("SOURCE:", food_source)
            print("RESULT:", result, confidence)

        except Exception as e:
            print("ERROR:", e)
            error = "Something went wrong."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error
    )

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
