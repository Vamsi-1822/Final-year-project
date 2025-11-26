import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

# ----------------------------------------------
# STREAMLIT PAGE SETUP
# ----------------------------------------------
st.set_page_config(page_title="Construction Optimization Dashboard", layout="wide")
st.title("🏗️ Construction Project Optimization Dashboard")
st.markdown("### Predict Risk Level, Best Cost, and Project Duration using Machine Learning")

# ----------------------------------------------
# UPLOAD OR USE SAMPLE DATASET
# ----------------------------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your Construction Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset Uploaded Successfully!")
else:
    st.sidebar.warning("Using Default Dataset (Construction_Dataset.csv)")
    df = pd.read_csv("Construction_Dataset.csv")

st.dataframe(df.head())

# ----------------------------------------------
# HELPER FUNCTION
# ----------------------------------------------
def regression_summary(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R² Score": r2}

# ----------------------------------------------
# CREATE TABS FOR EACH MODEL
# ----------------------------------------------
tabs = st.tabs(["🎯 Risk Level Prediction", "💰 Best Cost Prediction", "⏳ Project Duration Prediction"])

# =====================================================
# TAB 1 - RISK LEVEL PREDICTION (CLASSIFICATION)
# =====================================================
with tabs[0]:
    st.header("🎯 Risk Level Prediction")

    if "Risk Level" in df.columns:
        X = df.drop("Risk Level", axis=1)
        y = df["Risk Level"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Comparison Chart
        st.subheader("📊 Actual vs Predicted Risk Level")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_test.values[:30], label="Actual", marker="o")
        ax.plot(y_pred[:30], label="Predicted", marker="x")
        ax.legend()
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Risk Level")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔥 Feature Importance - Risk Level")
        importance = clf.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(X.columns, importance)
        st.pyplot(fig)

# =====================================================
# TAB 2 - BEST COST PREDICTION (REGRESSION)
# =====================================================
with tabs[1]:
    st.header("💰 Best Cost Prediction")

    if "Best Cost (BC)" in df.columns:
        X = df.drop("Best Cost (BC)", axis=1)
        y = df["Best Cost (BC)"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg_cost = RandomForestRegressor(random_state=42)
        reg_cost.fit(X_train, y_train)
        y_pred = reg_cost.predict(X_test)

        metrics = regression_summary(y_test, y_pred)
        st.subheader("📈 Model Performance Summary")
        st.dataframe(pd.DataFrame([metrics]))

        # Comparison Plot
        st.subheader("📊 Predicted vs Actual Best Cost")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Cost")
        ax.set_ylabel("Predicted Cost")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔥 Feature Importance - Best Cost (BC)")
        importance = reg_cost.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(X.columns, importance)
        st.pyplot(fig)

# =====================================================
# TAB 3 - PROJECT DURATION PREDICTION (REGRESSION)
# =====================================================
with tabs[2]:
    st.header("⏳ Project Duration Prediction")

    if "Project Duration (days)" in df.columns:
        X = df.drop("Project Duration (days)", axis=1)
        y = df["Project Duration (days)"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg_duration = RandomForestRegressor(random_state=42)
        reg_duration.fit(X_train, y_train)
        y_pred = reg_duration.predict(X_test)

        metrics = regression_summary(y_test, y_pred)
        st.subheader("📈 Model Performance Summary")
        st.dataframe(pd.DataFrame([metrics]))

        # Comparison Plot
        st.subheader("📊 Predicted vs Actual Duration")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Duration (days)")
        ax.set_ylabel("Predicted Duration (days)")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔥 Feature Importance - Project Duration")
        importance = reg_duration.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(X.columns, importance)
        st.pyplot(fig)

# =====================================================
# MANUAL INPUT PREDICTIONS (BOTTOM SECTION)
# =====================================================
st.markdown("---")
st.header("🧾 Predict for New Project Data")

cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
user_input = {col: st.number_input(f"Enter value for {col}", value=float(df[col].mean())) for col in cols}
input_df = pd.DataFrame([user_input])

st.write("### Your Input Data")
st.dataframe(input_df)

if st.button("🔮 Predict"):
    results = {}

    if "Risk Level" in df.columns:
        X_risk = df.drop("Risk Level", axis=1)
        input_risk = input_df[X_risk.columns]
        results["Risk Level"] = int(clf.predict(input_risk)[0])

    if "Best Cost (BC)" in df.columns:
        X_cost = df.drop("Best Cost (BC)", axis=1)
        input_cost = input_df[X_cost.columns]
        results["Best Cost (BC)"] = float(reg_cost.predict(input_cost)[0])

    if "Project Duration (days)" in df.columns:
        X_duration = df.drop("Project Duration (days)", axis=1)
        input_duration = input_df[X_duration.columns]
        results["Project Duration (days)"] = float(reg_duration.predict(input_duration)[0])

    st.success("### ✅ Prediction Results:")
    st.json(results)
