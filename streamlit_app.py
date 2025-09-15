import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="High-Risk User Review", layout="wide")

# -------------------------
# Upload
# -------------------------
st.title("🔎 High-Risk User Review Dashboard (One CSV)")
st.markdown("Upload the combined CSV (with all license, test, meta, and suspicious logs already joined).")

uploaded = st.file_uploader("Upload Combined Data CSV", type=["csv"])
if not uploaded:
    st.info("Please upload the combined CSV to proceed.")
    st.stop()

# -------------------------
# Read & Normalize
# -------------------------
df = pd.read_csv(uploaded)
df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

# parse date
if "date" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.date
elif "created_on" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["created_on"], errors="coerce").dt.date
else:
    df["date_parsed"] = pd.NaT

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters")
min_date, max_date = df["date_parsed"].min(), df["date_parsed"].max()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date))

user_list = sorted(df["user_id"].dropna().astype(str).unique())
user_select = st.sidebar.selectbox("User ID", options=[""] + user_list)

df_filt = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    df_filt = df_filt[(df_filt["date_parsed"] >= date_range[0]) & (df_filt["date_parsed"] <= date_range[1])]
if user_select:
    df_filt = df_filt[df_filt["user_id"].astype(str) == user_select]

# -------------------------
# Summary Table
# -------------------------
st.header("📊 Daily User-Device Summary")
summary_cols = [
    "date_parsed","user_id","device_id",
    "no_of_videos_hit_license_per_device",
    "number_of_hours_per_device_per_day",
    "count_of_gtr_than_90_videos_per_device",
    "qbank_completed_per_device",
    "test_completed_per_device",
    "cm_solve_per_device",
    "overall_account_non_video_activity_current_date_last_7_days_user_level",
    "bifurcation_pct_dev_levl_no_subject_filter",
    "total_no_of_videos_watched",
    "in_seq_watched",
    "sequence_distribution_sub_dwv_levl",
    "repeat_count_license_lvl_10",
    "repeat_count_hours_lvl_per_device_hours_consumption_4hrs",
    "subjects_accessed_per_device",
    "sub_count_per_device",
    "sub_completion_pct_per_device_overall_over_90_completion",
    "user_count_assoc_per_device_on_that_day",
    "limit_name_per_device_that_day",
    "dex_appearance_per_device_that_day"
]
summary_cols = [c for c in summary_cols if c in df_filt.columns]

st.dataframe(df_filt[summary_cols].head(200), use_container_width=True)

# -------------------------
# Visuals
# -------------------------
st.header("📈 Visuals")

# Videos hit per day
if "no_of_videos_hit_license_per_device" in df_filt.columns:
    daily_videos = df_filt.groupby("date_parsed")["no_of_videos_hit_license_per_device"].sum().reset_index()
    fig1 = px.bar(daily_videos, x="date_parsed", y="no_of_videos_hit_license_per_device", title="Total Videos Hit per Day")
    st.plotly_chart(fig1, use_container_width=True)

# Device share
if "device_id" in df_filt.columns and "no_of_videos_hit_license_per_device" in df_filt.columns:
    dev_share = df_filt.groupby("device_id")["no_of_videos_hit_license_per_device"].sum().reset_index()
    fig2 = px.pie(dev_share, names="device_id", values="no_of_videos_hit_license_per_device", title="Device Share (Videos)")
    st.plotly_chart(fig2, use_container_width=True)

# Risk Radar
if user_select:
    st.subheader(f"⚠️ Risk Profile for {user_select}")
    user_df = df[df["user_id"].astype(str) == user_select]
    if not user_df.empty:
        total_videos = user_df["no_of_videos_hit_license_per_device"].sum()
        devices = user_df["device_id"].nunique()
        days_active = user_df["date_parsed"].nunique()
        hours = user_df["number_of_hours_per_device_per_day"].sum() if "number_of_hours_per_device_per_day" in user_df else 0
        over90 = user_df["count_of_gtr_than_90_videos_per_device"].sum() if "count_of_gtr_than_90_videos_per_device" in user_df else 0

        vals = [
            min(total_videos/200,1),
            min(devices/5,1),
            min(days_active/30,1),
            min(hours/50,1),
            min(over90/100,1)
        ]
        labels = ["Videos","Devices","Days Active","Hours",">90% Completed"]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=vals, theta=labels, fill="toself"))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False)
        st.plotly_chart(radar, use_container_width=True)

# -------------------------
# Downloads
# -------------------------
st.header("⬇️ Downloads")
st.download_button("Download Filtered Data", df_filt.to_csv(index=False).encode("utf-8"), "filtered_data.csv")

st.markdown("---")
st.caption("✅ One-file version: All KPIs and visuals are driven from the same uploaded CSV.")
