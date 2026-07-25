# ==========================================
# Diabetes Prediction
# Using Saved Random Forest Model
# ==========================================

import pandas as pd
import joblib


# ==========================================
# Load Model
# ==========================================

MODEL_PATH = r"models/trained_model.pkl"

model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully!")


# Get training feature order
model_features = model.feature_names_in_

print("\nModel Features:")
print(model_features)



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
        diabetes_pedigree_function,
        age
):


    # ======================================
    # Create original dataframe
    # ======================================

    df = pd.DataFrame({

        "Pregnancies":[pregnancies],

        "Glucose":[glucose],

        "BloodPressure":[blood_pressure],

        "SkinThickness":[skin_thickness],

        "Insulin":[insulin],

        "BMI":[bmi],

        "DiabetesPedigreeFunction":[
            diabetes_pedigree_function
        ],

        "Age":[age]

    })



    # ======================================
    # Replace zeros with NaN
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


    # Since prediction is one patient,
    # use median values from training data logic
    # Here we replace missing with 0
    # (better: save training medians)

    df = df.fillna(0)



    # ======================================
    # Feature Engineering
    # ======================================


    # -------- BMI --------

    if df["BMI"].iloc[0] < 18.5:

        df["NEW_BMI_CAT"] = "underweight"

    elif df["BMI"].iloc[0] < 25:

        df["NEW_BMI_CAT"] = "normal"

    elif df["BMI"].iloc[0] < 30:

        df["NEW_BMI_CAT"] = "overweight"

    else:

        df["NEW_BMI_CAT"] = "obese"



    df.drop(
        "BMI",
        axis=1,
        inplace=True
    )



    # -------- Glucose --------

    if glucose < 70:

        df["NEW_GLUCOSE_CAT"] = "low"

    elif glucose < 99:

        df["NEW_GLUCOSE_CAT"] = "normal"

    elif glucose < 126:

        df["NEW_GLUCOSE_CAT"] = "high"

    else:

        df["NEW_GLUCOSE_CAT"] = "very_high"



    # -------- Skin Thickness --------

    if skin_thickness < 30:

        df["NEW_SKIN_THICKNESS"] = "normal"

    elif skin_thickness >= 70:

        df["NEW_SKIN_THICKNESS"] = "highfat"



    # -------- Pregnancy --------

    if pregnancies == 0:

        df["NEW_PREGNANCIES"] = "no_pregnancies"

    elif pregnancies <=4:

        df["NEW_PREGNANCIES"] = "std_pregnancies"

    else:

        df["NEW_PREGNANCIES"] = "over_pregnancies"



    # -------- Circulation --------

    if skin_thickness < 30 and blood_pressure < 80:

        df["NEW_CIRCULATION_LEVEL"] = "normal"

    elif skin_thickness >=30 and blood_pressure >=80:

        df["NEW_CIRCULATION_LEVEL"] = "high_risk"

    else:

        df["NEW_CIRCULATION_LEVEL"] = "medium_risk"



    # -------- New features --------

    df["PRE_AGE_CAT"] = pregnancies * age


    df["INSULIN_GLUCOSE_CAT"] = (
        insulin * glucose
    )



    # Drop same columns as training

    df.drop(
        ["Pregnancies","Glucose","SkinThickness"],
        axis=1,
        inplace=True
    )



    # ======================================
    # One Hot Encoding
    # ======================================


    df = pd.get_dummies(df)



    # ======================================
    # Match training columns
    # ======================================

    df = df.reindex(
        columns=model_features,
        fill_value=0
    )



    # ======================================
    # Prediction
    # ======================================


    prediction = model.predict(df)[0]


    probability = model.predict_proba(df)[0]


    return prediction, probability




# ==========================================
# Test Patient
# ==========================================


result, probability = predict_diabetes(

    pregnancies=2,

    glucose=150,

    blood_pressure=80,

    skin_thickness=35,

    insulin=130,

    bmi=32,

    diabetes_pedigree_function=0.45,

    age=45

)



# ==========================================
# Result
# ==========================================


if result == 1:

    print("\n🩺 Prediction: Diabetic")

else:

    print("\n✅ Prediction: Healthy")


print(
    f"Healthy Probability: {probability[0]*100:.2f}%"
)


print(
    f"Diabetic Probability: {probability[1]*100:.2f}%"
)