import json

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Sydney House Price Predictor", page_icon="🏠")

@st.cache_resource
def load_assets():
    model = XGBRegressor()
    model.load_model("xgb_model.json")
    with open("feature_names.json") as f:
        features = json.load(f)
    with open("defaults.json") as f:
        defaults = json.load(f)
    suburbs = pd.read_csv("suburb_lookup.csv").set_index("suburb")
    return model, features, defaults, suburbs


model, features, defaults, suburbs = load_assets()

st.title("🏠 Sydney House Price Predictor")
st.write("Estimate a sale price using an XGBoost model trained on Sydney property sales.")

col1, col2 = st.columns(2)
with col1:
    suburb = st.selectbox("Suburb", sorted(suburbs.index))
    types = [c[len("type_"):] for c in features if c.startswith("type_")]
    default_type = types.index("House") if "House" in types else 0
    prop_type = st.selectbox("Property type", types, index=default_type)
    size = st.number_input("Property size (sqm)", min_value=20, max_value=5000, value=600)
with col2:
    beds = st.slider("Bedrooms", 1, 10, 3)
    baths = st.slider("Bathrooms", 1, 8, 2)
    parking = st.slider("Parking spaces", 0, 8, 2)
    month = st.slider("Month of sale", 1, 12, 6)
    year = st.number_input("Year of sale", min_value=2016, max_value=2022,
                           value=int(defaults["year_sold"]))

with st.expander("Market assumptions (advanced)"):
    cash_rate = st.number_input("Cash rate (%)", value=float(defaults["cash_rate"]))
    infl = st.number_input(
        "Property inflation index", value=float(defaults["property_inflation_index"])
    )

# Build one row in exactly the column order the model was trained on
row = {f: 0.0 for f in features}
row.update(suburbs.loc[suburb].to_dict())
row.update(
    {
        "num_bed": beds,
        "num_bath": baths,
        "num_parking": parking,
        "property_size": size,
        "month_sold": month,
        "year_sold": year,
        "cash_rate": cash_rate,
        "property_inflation_index": infl,
    }
)
if f"type_{prop_type}" in row:
    row[f"type_{prop_type}"] = 1.0

X_new = pd.DataFrame([row])[features].astype(float)

if st.button("Predict price"):
    pred = 10 ** model.predict(X_new)[0]  # target was log10(price)
    band = defaults["error_factor"]  # typical test error as a multiplier
    st.metric("Estimated price", f"${pred:,.0f}")
    st.caption(f"Typical range: ${pred / band:,.0f} to ${pred * band:,.0f}")
    st.caption("Model trained on Sydney sales from 2016 to January 2022, so estimates reflect that market.")
    st.caption("For educational use only. Not financial advice.")
    st.caption("By Chitresh Berwal")