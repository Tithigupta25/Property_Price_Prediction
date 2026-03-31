from fastapi import FastAPI
import pandas as pd
import joblib

model = joblib.load("pkl/model.pkl")
le = joblib.load("pkl/encoder.pkl")
avg_error = joblib.load("pkl/avg_error.pkl")
feature = joblib.load("pkl/features.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {  "message": "Property Price Prediction API 🚀",
        "required_features": [
            "area",
            "bedrooms",
            "location",
            "amenities_score",
            "project_count",
            "news_sentiment"
        ],
        "note": "Send POST request to /predict with above features"
    }

@app.post("/predict")
def predict(data: dict):
    try:
        input_dict = {
            "area": data.get("area", 0),
            "location": data.get("location", ""),
            "bedrooms": data.get("bedrooms", 0),
            "amenities_score": data.get("amenities_score", 0),
            "project_count": data.get("project_count", 0),
            "news_sentiment": data.get("news_sentiment", 0)
        }

        input_data = pd.DataFrame([input_dict])

        input_data["location"] = input_data["location"].astype(str).str.strip()

        try:
            encoded_val = le.transform([input_data.loc[0, "location"]])[0]
        except:
            encoded_val = 0

        input_data.loc[0, "location"] = int(encoded_val)

        input_data = input_data.reindex(columns=feature, fill_value=0)
        input_data = input_data.astype(float)
        prediction = model.predict(input_data)[0]

        if prediction != 0:
            confidence = 1 / (1 + (avg_error / prediction))
        else:
            confidence = 0

        confidence = max(0, min(confidence, 1))

        try:
            importances = model.feature_importances_
            feature_importance = pd.Series(importances, index=feature)
            top_feature = (feature_importance.sort_values(ascending=False).head(3).index.tolist())
            
        except:
            top_feature = ["area", "location", "amenities_score"]

        return {
            "predicted_price": int(prediction),
            "confidence_score": round(confidence, 2),
            "top_factors": top_feature
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Something went wrong"
        }