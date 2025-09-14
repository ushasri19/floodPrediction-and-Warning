import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, scaler, and feature list
model = joblib.load("flood_final_model.pkl")
scaler = joblib.load("flood_scaler.pkl")
feature_list = joblib.load("feature_list.pkl")

st.set_page_config(
    page_title="🌊 Flood Risk Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with input sliders
st.sidebar.image(
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=60",
    caption="Climate-induced Flood Risk",
    use_container_width=True
)
st.sidebar.header("Input Flood Risk Factors")

user_data = {}
for feature in feature_list:
    user_data[feature] = st.sidebar.slider(feature, 0, 10, 5)

st.sidebar.markdown("""
---
Adjust the sliders to set environmental and infrastructural factors.
Click **Predict Flood Risk** to see the forecasted category.
""")

if st.sidebar.button("Predict Flood Risk"):
    input_df = pd.DataFrame([user_data])[feature_list]
    st.write("Input DataFrame:", input_df)                      # Debug: show inputs
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)
    st.write("Prediction probabilities:", prediction_proba)    # Debug: show class probabilities
    st.sidebar.success(f"🚩 Predicted Flood Risk Category: **{prediction}**")
    st.balloons()
else:
    st.sidebar.info("Set flood risk factors using the sliders and press the button above.")

# Main page content
st.title("🌊 Flood Risk Prediction App")
st.markdown("""
Welcome to the Flood Risk Prediction application. This tool estimates flood risk levels 
based on key climate, geographical, and infrastructural factors using a Random Forest model.
""")




with st.expander("🧐 What is Flood Risk and Why It Matters?"):
    st.write("""
    Flood risk is the potential for flooding in a given area, influenced by weather conditions, land use, 
    and human activities. Timely prediction helps in mitigation and saves lives.
    """)
    st.write("This app uses machine learning with environmental data to predict these risks.")

st.header("Flood Risk Levels")
col_low, col_medium, col_high = st.columns(3)

col_low.markdown("### 🟢 Low Risk\nMinimal flood probability.\nNormal precautions recommended.")
col_medium.markdown("### 🟠 Medium Risk\nModerate flood chance.\nBe alert and monitor local advisories.")
col_high.markdown("### 🔴 High Risk\nSevere flood threat.\nPrepare and follow emergency protocols.")

if st.checkbox("Show Model Feature Importance"):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('Most Influential Features in Flood Risk Prediction')
    st.pyplot(plt.gcf())

st.markdown("---")
st.caption("Developed using Python, Streamlit, and Random Forest model trained on climate & infrastructure data.")
