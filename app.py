import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Video Usage Dashboard", layout="wide")

st.title("📊 Video Usage & Suspicious Activity Dashboard")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload joined dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Convert dates ---
    for col in ["created_on", "submitted_on", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", unit="ms", origin="unix")
    
    # --- Filters ---
    users = df["user_id"].dropna().unique()
    selected_user = st.sidebar.selectbox("Select User", options=["All"]+list(users))
    date_min, date_max = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Date Range", [date_min, date_max])
    
    # Apply filters
    mask = (df["date"].between(date_range[0], date_range[1]))
    if selected_user != "All":
        mask &= (df["user_id"] == selected_user)
    fdf = df[mask]
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["License", "Lesson Tests", "Suspicious Logs", "Video Meta"])
    
    with tab1:
        st.subheader("License Metrics")
        license_df = fdf.groupby(["user_id","device_id","date"]).agg(
            license_count=("lesson_id","nunique"),
            subjects=(" _subject_title","nunique")
        ).reset_index()
        
        license_df["repeat_day"] = np.where(license_df["license_count"] > 10, 1, 0)
        
        st.dataframe(license_df)
        st.plotly_chart(px.bar(license_df, x="date", y="license_count", color="device_id"))
    
    with tab2:
        st.subheader("Lesson Test Submissions")
        submissions = fdf.groupby(["user_id","device_id","date"]).agg(
            qbank_submits=("content_id", lambda x: x[fdf["content_type"]=="lesson"].nunique()),
            test_submits=("content_id", lambda x: x[fdf["content_type"]=="test"].nunique())
        ).reset_index()
        
        st.dataframe(submissions)
        st.plotly_chart(px.bar(submissions, x="date", y=["qbank_submits","test_submits"]))
    
    with tab3:
        st.subheader("Suspicious Activity Logs")
        susp = fdf[["user_id","category","sub_category","alert_level","alert_type","message","date"]]
        st.dataframe(susp)
        st.plotly_chart(px.histogram(susp, x="date", color="alert_level"))
    
    with tab4:
        st.subheader("Video Meta")
        meta = fdf.groupby("_subject_title").agg(lessons=("lesson_id","nunique")).reset_index()
        st.dataframe(meta)
        st.plotly_chart(px.pie(meta, names="_subject_title", values="lessons"))
