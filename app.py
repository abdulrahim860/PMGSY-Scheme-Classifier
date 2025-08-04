import streamlit as st
import pandas as pd
import joblib
from feature_engineer import FeatureEngineer
from xgboost import XGBClassifier

preprocessor = joblib.load("pmgsy_preprocessor.pkl")
le_scheme=joblib.load("scheme_label_encoder.pkl")

xgb = XGBClassifier()
xgb.load_model("pmgsy_xgb.json")

st.title("🛣️ PMGSY Scheme Classifier")
st.write("Upload Project data(csv)")

uploaded_file=st.file_uploader("Upload CSV file",type="csv")

if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.subheader("Uploaded data")
    st.dataframe(data.head())
    if "PMGSY_SCHEME" in data.columns:
        data=data.drop("PMGSY_SCHEME",axis=1)
    fe = FeatureEngineer()
    data = fe.transform(data)

    X_processed = preprocessor.transform(data)

    predictions_num = xgb.predict(X_processed)
    predictions = le_scheme.inverse_transform(predictions_num)
    data["Predicted_Scheme"]=predictions
    st.subheader("predicted Schemes")
    st.dataframe(data)

    csv=data.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download predictions",csv,"pmgsy_predictions.csv","text/csv") 