from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


# ==========================================
# Load Model and Encoder
# ==========================================

MODEL_PATH = r"models/trained_model.pkl"

ENCODER_PATH = r"models/label_encoder.pkl"


model = joblib.load(MODEL_PATH)

encoder = joblib.load(ENCODER_PATH)


print("✅ Model loaded successfully!")
print("✅ Encoder loaded successfully!")


# Features used during training

model_features = model.feature_names_in_



# ==========================================
# Prediction Function
# ==========================================

def predict_diabetes(
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
):


    # Create input dataframe

    df = pd.DataFrame({

        "Pregnancies":[pregnancies],

        "Glucose":[glucose],

        "BloodPressure":[blood_pressure],

        "SkinThickness":[skin_thickness],

        "Insulin":[insulin],

        "BMI":[bmi],

        "DiabetesPedigreeFunction":[
            diabetes_pedigree
        ],

        "Age":[age]

    })



    # ======================================
    # Replace zero values
    # ======================================

    cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]


    df[cols] = df[cols].replace(
        0,
        pd.NA
    )


    df = df.fillna(0)



    # ======================================
    # BMI Feature
    # ======================================


    if bmi < 18.5:

        df["NEW_BMI_CAT"] = "underweight"

    elif bmi < 25:

        df["NEW_BMI_CAT"] = "normal"

    elif bmi < 30:

        df["NEW_BMI_CAT"] = "overweight"

    else:

        df["NEW_BMI_CAT"] = "obese"



    df.drop(
        "BMI",
        axis=1,
        inplace=True
    )



    # ======================================
    # Glucose Feature
    # ======================================


    if glucose < 70:

        df["NEW_GLUCOSE_CAT"] = "low"

    elif glucose < 99:

        df["NEW_GLUCOSE_CAT"] = "normal"

    elif glucose < 126:

        df["NEW_GLUCOSE_CAT"] = "high"

    else:

        df["NEW_GLUCOSE_CAT"] = "very_high"




    # ======================================
    # Skin Thickness Feature
    # ======================================


    if skin_thickness < 30:

        df["NEW_SKIN_THICKNESS"] = "normal"

    elif skin_thickness >= 70:

        df["NEW_SKIN_THICKNESS"] = "highfat"




    # ======================================
    # Pregnancy Feature
    # ======================================


    if pregnancies == 0:

        df["NEW_PREGNANCIES"] = "no_pregnancies"


    elif pregnancies <= 4:

        df["NEW_PREGNANCIES"] = "std_pregnancies"


    else:

        df["NEW_PREGNANCIES"] = "over_pregnancies"




    # ======================================
    # Circulation Feature
    # ======================================


    if skin_thickness < 30 and blood_pressure < 80:

        df["NEW_CIRCULATION_LEVEL"] = "normal"


    elif skin_thickness >= 30 and blood_pressure >= 80:

        df["NEW_CIRCULATION_LEVEL"] = "high_risk"


    else:

        df["NEW_CIRCULATION_LEVEL"] = "medium_risk"




    # ======================================
    # New Features
    # ======================================


    df["PRE_AGE_CAT"] = (
        pregnancies * age
    )


    df["INSULIN_GLUCOSE_CAT"] = (
        insulin * glucose
    )




    # Same dropping as training

    df.drop(
        [
            "Pregnancies",
            "Glucose",
            "SkinThickness"
        ],
        axis=1,
        inplace=True
    )



    # One hot encoding

    df = pd.get_dummies(df)



    # Match training columns

    df = df.reindex(
        columns=model_features,
        fill_value=0
    )



    # Prediction

    prediction = model.predict(df)[0]


    probability = model.predict_proba(df)[0]


    return prediction, probability




# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )




# ==========================================
# Predict Route
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():


    prediction, probability = predict_diabetes(

        int(request.form["pregnancies"]),

        float(request.form["glucose"]),

        float(request.form["blood_pressure"]),

        float(request.form["skin_thickness"]),

        float(request.form["insulin"]),

        float(request.form["bmi"]),

        float(request.form["diabetes_pedigree"]),

        int(request.form["age"])

    )



    if prediction == 1:

        result = "🩺 Diabetic"

    else:

        result = "✅ Healthy"




    return render_template(

        "index.html",

        prediction=result,

        healthy=round(
            probability[0]*100,
            2
        ),

        diabetic=round(
            probability[1]*100,
            2
        )

    )




# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
        use_reloader=False
    )