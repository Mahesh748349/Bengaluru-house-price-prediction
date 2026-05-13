import joblib
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, TRAIN_PATH
from src.feature_engineering import add_real_estate_features, add_train_based_location_features
from src.modeling import predict_prices
from src.preprocessing import basic_clean

MODEL_PATH = MODELS_DIR / "best_model.joblib"


@st.cache_data(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_train_data():
    df = pd.read_csv(TRAIN_PATH)
    return df


def prepare_input_dataframe(input_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    input_df = basic_clean(input_df)
    input_df = add_real_estate_features(input_df)

    train_df = basic_clean(train_df)
    train_df = add_real_estate_features(train_df)
    _, input_df = add_train_based_location_features(train_df, input_df, target_col="price")

    return input_df


def main():
    st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")
    st.title("🏠 Bangalore House Price Predictor")
    st.write("Enter a property profile and get a predicted sale price based on the project model.")

    train_df = load_train_data()
    model = load_model()

    with st.form("prediction_form"):
        area = st.number_input("Area (sqft)", min_value=100.0, max_value=20000.0, value=1200.0, step=10.0)
        location = st.text_input("Location", value="Sarjapur Road")
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
        bath = st.number_input("Bathroom count", min_value=1, max_value=10, value=2, step=1)
        balcony = st.number_input("Balcony count", min_value=0, max_value=5, value=1, step=1)
        parking = st.number_input("Parking spaces", min_value=0, max_value=5, value=1, step=1)

        furnishing_options = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
        furnishing = st.selectbox("Furnishing", furnishing_options, index=0)

        property_type_options = ["Apartment", "Independent House", "Villa", "Other"]
        property_type = st.selectbox("Property Type", property_type_options, index=0)

        age = st.number_input("Age of property (years)", min_value=0, max_value=100, value=5, step=1)

        submit_button = st.form_submit_button("Predict Price")

    if submit_button:
        input_df = pd.DataFrame(
            [
                {
                    "area": area,
                    "location": location,
                    "bhk": float(bhk),
                    "bath": float(bath),
                    "balcony": int(balcony),
                    "parking": int(parking),
                    "furnishing": furnishing,
                    "property_type": property_type,
                    "age": int(age),
                }
            ]
        )
        input_df = prepare_input_dataframe(input_df, train_df)
        prediction = predict_prices(model, input_df)

        if len(prediction) == 1:
            predicted_price = prediction[0]
            
            # Estimate confidence range based on price tier
            if predicted_price < 5e6:
                margin_pct = 0.44  # 44% error for < 50L
            elif predicted_price < 10e6:
                margin_pct = 0.30  # 30% error for 50L-1Cr
            elif predicted_price < 15e6:
                margin_pct = 0.20  # 20% error for 1Cr-1.5Cr
            elif predicted_price < 25e6:
                margin_pct = 0.19  # 19% error for 1.5Cr-2.5Cr
            else:
                margin_pct = 0.28  # 28% error for > 2.5Cr
            
            lower_bound = predicted_price * (1 - margin_pct)
            upper_bound = predicted_price * (1 + margin_pct)
            
            price_text = f"₹{predicted_price:,.0f}"
            st.success("Prediction ready")
            st.metric("Predicted sale price", price_text)
            
            st.info(
                f"📊 **Prediction Range:** ₹{lower_bound:,.0f} to ₹{upper_bound:,.0f}\n\n"
                f"*Based on historical accuracy (~{margin_pct*100:.0f}% average error on similar properties)*"
            )
        else:
            st.error("Prediction failed. Please try again with valid inputs.")

    st.markdown("---")
    st.markdown("### 📋 How Accurate Is This?")
    st.markdown(
        """
- **Model accuracy:** ~20-30% average error on mid-range properties (₹1-2.5Cr)
- **Lower on extremes:** Budget homes (< 50L) and ultra-premium (> 2.5Cr) have ~30-45% error
- **Why?** Limited training examples for rare price tiers + Bengaluru's complex location premiums
- **Use as:** Market reference point + negotiation baseline, not absolute truth
- **Best for:** Screening properties in ₹1-2.5Cr range
        """
    )
    
    st.markdown("---")
    st.markdown("### Notes")
    st.markdown(
        "- The app uses the trained model file `models/best_model.joblib` and the dataset in `data/train.csv`."
    )
    st.markdown(
        "- If you change `data/train.csv`, re-run `python -m src.run_pipeline` before using the app again."
    )
    st.markdown(
        "- This UI is built with Streamlit for a quick website-style prediction interface."
    )


if __name__ == "__main__":
    main()
