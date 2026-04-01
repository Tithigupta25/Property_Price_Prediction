import streamlit as st
import pandas as pd
import joblib

model = joblib.load("pkl/model.pkl")
le = joblib.load("pkl/encoder.pkl")
avg_error = joblib.load("pkl/avg_error.pkl")
features = joblib.load("pkl/features.pkl")
location_list = list(le.classes_)

st.set_page_config(page_title="Property Price Predictor")

st.title("Property Price Prediction")
st.write("Enter property details to estimate price")

area = st.number_input("Area (sqft)", 0)
bedrooms = st.number_input("Bedrooms", 0)
location = st.selectbox("Select Location", location_list)

amenities_score = st.number_input("Amenities Score", 0)
project_count = st.number_input("Nearby Projects Count", 0)
news_sentiment = st.number_input("News Sentiment (-1 to 1)", 0.0)

if st.button("Predict Price"):

    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "location": location.strip(),
        "amenities_score": amenities_score,
        "project_count": project_count,
        "news_sentiment": news_sentiment
    }

    input_df = pd.DataFrame([input_dict])

    try:
        input_df["location"] = le.transform([input_df.loc[0, "location"]])[0]
    except:
        st.warning("Unknown location — using default encoding")
        input_df["location"] = 0

    input_df = input_df.reindex(columns=features, fill_value=0)
    input_df = input_df.astype(float)

    prediction = model.predict(input_df)[0]

    if prediction != 0:
        confidence = 1 / (1 + (avg_error / prediction))
    else:
        confidence = 0

    confidence = max(0, min(confidence, 1))

    try:
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=features)
        top_features = feature_importance.sort_values(ascending=False).head(3).index.tolist()
    except:
        top_features = ["area", "location", "amenities_score"]

    st.success(f"Predicted Price: ₹ {int(prediction)}")
    st.info(f"Confidence Score: {round(confidence, 2)}")
    st.write("Top Influencing Factors:", top_features)