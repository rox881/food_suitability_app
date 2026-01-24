from flask import Flask, render_template, request
from utils.food_service import get_food_nutrition
from utils.ml_service import predict_suitability

# ----------------------------
# App initialization
# ----------------------------
app = Flask(__name__)

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
# Route
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.json if request.is_json else request.form
        
        try:
            # Step 1: Validate user
            user, error = get_user_input(data)
            if error:
                return _response(error=error)

            # Step 2: Resolve food
            food_name = data.get("food", "")
            food, food_source, match_name = get_food_nutrition(food_name)

            if food is None:
                return _response(error="Food not found. Try 'Apple', 'Rice', etc.")

            # Step 3: Hybrid Prediction (ML + Rules)
            result, confidence, override_reason = predict_suitability(user, food, food_source)
            
            # Log for debugging
            print(f"User: {user}, Food: {match_name}, Result: {result}, Reason: {override_reason}")

            return _response(
                result=result, 
                confidence=confidence, 
                food_matched=match_name,
                source=food_source,
                reason=override_reason
            )

        except Exception as e:
            print("ERROR:", e)
            return _response(error="Something went wrong.")

    return render_template("index.html")

def _response(result=None, confidence=None, error=None, food_matched=None, source=None, reason=None):
    if request.is_json:
        return {
            "result": result,
            "confidence": confidence,
            "error": error,
            "food": food_matched,
            "source": source,
            "reason": reason
        }
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error,
        food_matched=food_matched
    )

if __name__ == "__main__":
    app.run(debug=True)
