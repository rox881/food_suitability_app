from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# App initialization
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Load model (absolute path)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "random_forest_food_suitability_model_v1.pkl"
)

model = joblib.load(MODEL_PATH)

# ----------------------------
# Load food dataset
# ----------------------------
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset_B_cleaned.csv"
)

food_df = pd.read_csv(DATA_PATH)

# Normalize food names once
food_df["Food"] = (
    food_df["Food"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ----------------------------
# Function 1: Get & validate user input
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
# Function 2: Get food nutrition (ROBUST)
# ----------------------------
def get_food_nutrition(food_name):
    food_name = food_name.strip().lower()

    if not food_name:
        return None

    # Partial + case-insensitive match
    matches = food_df[
        food_df["Food"].str.contains(food_name, case=False, na=False)
    ]

    if matches.empty:
        return None

    # Take the first best match
    row = matches.iloc[0]

    return {
        "calories": float(row["Calories"]),
        "protein": float(row["Protein"]),
        "fat": float(row["Fat"]),
        "carbs": float(row["Carbohydrates"])
    }


# ----------------------------
# Function 3: Create model input
# ----------------------------
def create_model_input(user, food):
    return np.array([[
        user["weight"],
        user["bmi"],
        food["calories"],
        food["protein"],
        food["fat"],
        food["carbs"]
    ]])


# ----------------------------
# Route
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            # Step 1: Validate user input
            user, error = get_user_input(request.form)
            if error:
                return render_template("index.html", error=error)

            # Step 2: Fetch food nutrition
            food_name = request.form.get("food", "")
            food = get_food_nutrition(food_name)

            if food is None:
                return render_template(
                    "index.html",
                    error="Food not found in database."
                )

            # Step 3: Create model input
            X = create_model_input(user, food)

            # Step 4: Model prediction
            pred = int(model.predict(X)[0])
            conf = float(model.predict_proba(X)[0][pred])

            result = "Suitable" if pred == 1 else "Not Suitable"
            confidence = round(conf, 3)

            # Debug logs (safe)
            print("USER:", user)
            print("FOOD INPUT:", food_name)
            print("FOOD USED:", food)
            print("PREDICTION:", result, confidence)

        except Exception as e:
            print("ERROR:", e)
            error = "Something went wrong. Please try again."

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
