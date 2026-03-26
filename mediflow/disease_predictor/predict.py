import joblib
import numpy as np

model = joblib.load("disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")
le = joblib.load("disease_encoder.pkl")

# Example symptoms
symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions"
]

# Encode
input_data = mlb.transform([symptoms])

# Predict
prediction = model.predict(input_data)

disease = le.inverse_transform(prediction)

print("Predicted Disease:", disease[0])