import streamlit as st
import pandas as pd
from pmgsy.model_utils import load_models, predict_scheme, predict_cost

clf_pipeline, reg_pipeline, le_scheme = load_models()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Scheme Classifier", "Cost Prediction"])

if "last_page" not in st.session_state:
    st.session_state.last_page = page
    
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = "file_uploader"

if page != st.session_state.last_page:
    st.session_state.file_uploader_key = str(pd.Timestamp.now())
    st.session_state.last_page = page

st.title("🛣️ PMGSY Scheme Classifier")
st.write("Upload Project data(csv)")

uploaded_file=st.file_uploader("Upload CSV file",type="csv",key=st.session_state.file_uploader_key)

if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.subheader("Uploaded data")
    st.dataframe(data)

    original_data=data.copy()

    if page == "Scheme Classifier":
        if "PMGSY_SCHEME" in data.columns:
            data = data.drop("PMGSY_SCHEME", axis=1)

        st.markdown("## 🔍 Scheme Classification")
        predictions = predict_scheme(clf_pipeline, le_scheme, data)
        original_data["Predicted_Scheme"] = predictions
        st.dataframe(original_data[["Predicted_Scheme"]])

    elif page == "Cost Prediction":
        if "COST_OF_WORKS_SANCTIONED" in data.columns:
            data = data.drop("COST_OF_WORKS_SANCTIONED", axis=1)

        st.markdown("## 💰 Estimated Funding")
        estimated_costs = predict_cost(reg_pipeline, data)
        original_data["Estimated_Funding"] = estimated_costs.round(2)
        st.dataframe(original_data[["Estimated_Funding"]])

    st.markdown("## 📥 Download Full Results")
    results_csv = original_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Predictions",
        results_csv,
        "pmgsy_predictions.csv",
        "text/csv"
    )

else:
    st.info("👆 Please upload a CSV file to begin.") 