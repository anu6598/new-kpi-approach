import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="High-Risk User Review", layout="wide")

# -------------------------
# Helpers
# -------------------------

def parse_date_str(d):
    """Try to parse date-like strings (YYYY/MM/DD or YYYY-MM-DD) to date."""
    if pd.isna(d):
        return pd.NaT
    try:
        return pd.to_datetime(d, errors='coerce').dt.date if isinstance(d, pd.Series) else pd.to_datetime(d).date()
    except Exception:
        try:
            return pd.to_datetime(d, format='%Y/%m/%d', errors='coerce').date()
        except Exception:
            return pd.NaT


def millis_to_datetime(col):
    # col may be numeric or string
    return pd.to_datetime(col.astype('float64') / 1000, unit='s', errors='coerce')


def safe_lower(x):
    return str(x).strip().lower() if pd.notna(x) else ''


# -------------------------
# Upload area
# -------------------------
st.title("🔎 High-Risk User Review Dashboard (one-pager)")
st.markdown("Upload the three CSVs (license, lesson_test_submissions, video_meta) and optionally suspicious_activity_logs. Files must be per-day exports or full history.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    f_license = st.file_uploader("Upload video_license_logs CSV", type=["csv"], key='license')
with col2:
    f_lesson = st.file_uploader("Upload lesson_test_submissions CSV", type=["csv"], key='lesson')
with col3:
    f_meta = st.file_uploader("Upload video_meta CSV (optional)", type=["csv"], key='meta')
with col4:
    f_susp = st.file_uploader("Upload suspicious_activity_logs CSV (optional)", type=["csv"], key='susp')

if not f_license:
    st.info("Please upload video_license_logs CSV to proceed.")
    st.stop()

# read license
license_df = pd.read_csv(f_license)
# normalize column names
license_df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in license_df.columns]

# ensure date column exists and is parsed
if 'date' in license_df.columns:
    # some date strings are like 2025/09/11
    try:
        license_df['date_parsed'] = pd.to_datetime(license_df['date'], errors='coerce').dt.date
    except Exception:
        license_df['date_parsed'] = license_df['date'].apply(lambda x: pd.to_datetime(x, errors='coerce')).dt.date
else:
    # try created_on
    if 'created_on' in license_df.columns:
        license_df['date_parsed'] = pd.to_datetime(license_df['created_on'], errors='coerce').dt.date
    else:
        license_df['date_parsed'] = pd.NaT

# basic normalization for expected columns
for col in ['user_id','device_id','lesson_id','lesson_id','edition','ip','x_real_ip']:
    if col not in license_df.columns:
        license_df[col] = np.nan

# read lesson tests
if f_lesson:
    ldf = pd.read_csv(f_lesson)
    ldf.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in ldf.columns]
    # submitted_on is milliseconds string
    if 'submitted_on' in ldf.columns:
        ldf['submitted_on_dt'] = millis_to_datetime(ldf['submitted_on'])
        ldf['submit_date'] = ldf['submitted_on_dt'].dt.date
    else:
        ldf['submitted_on_dt'] = pd.NaT
        ldf['submit_date'] = pd.NaT
else:
    ldf = pd.DataFrame()

# read video meta
if f_meta:
    meta = pd.read_csv(f_meta)
    meta.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in meta.columns]
else:
    meta = pd.DataFrame()

# suspicious logs
if f_susp:
    susp = pd.read_csv(f_susp)
    susp.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in susp.columns]
else:
    susp = pd.DataFrame()

# -------------------------
# UI filters
# -------------------------
min_date = license_df['date_parsed'].min()
max_date = license_df['date_parsed'].max()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date range (license table)", value=(min_date, max_date) if pd.notna(min_date) else None)
user_filter = st.sidebar.text_input("User ID (optional)")
selected_user = None

# build user list from license and lesson tables
user_list = pd.unique(np.concatenate([
    license_df['user_id'].dropna().astype(str).unique(),
    ldf['user_id'].dropna().astype(str).unique() if not ldf.empty else np.array([])
]))
user_list = [u for u in user_list if u and u!='nan']
user_list_sorted = sorted(user_list)
user_select = st.sidebar.selectbox("Quick select user (optional)", options=[''] + user_list_sorted)

if user_select:
    user_filter = user_select

# apply date filter to license
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    license_filtered = license_df[(license_df['date_parsed'] >= start_d) & (license_df['date_parsed'] <= end_d)]
else:
    license_filtered = license_df.copy()

# apply user filter
if user_filter:
    license_filtered = license_filtered[license_filtered['user_id'].astype(str) == str(user_filter)]

# -------------------------
# License-level aggregations per device per day
# -------------------------
st.header("License: device/day level summary")

# compute videos hit license per device (count distinct lesson_id)
license_agg = (
    license_filtered.groupby(['date_parsed','user_id','device_id'], dropna=False)
    .agg(
        videos_hit_license=('lesson_id', lambda s: s.dropna().astype(str).nunique()),
        total_rows=('lesson_id', 'count')
    )
    .reset_index()
)

# subjects per device using meta if available
if not meta.empty and 'lesson_id' in meta.columns:
    meta_small = meta[['lesson_id','_subject_title']].drop_duplicates()
    # join to license_filtered
    lf = license_filtered.merge(meta_small, left_on='lesson_id', right_on='lesson_id', how='left')
    subj = (lf.groupby(['date_parsed','user_id','device_id'])['_subject_title']
            .apply(lambda s: ', '.join(sorted(set([x for x in s.dropna().astype(str)]))))
            .reset_index(name='subjects_per_device'))
    license_agg = license_agg.merge(subj, on=['date_parsed','user_id','device_id'], how='left')
else:
    license_agg['subjects_per_device'] = ''

# repeat count license: per user count distinct dates where license count > 10
user_repeat = (
    license_filtered.groupby(['date_parsed','user_id'])['lesson_id']
    .apply(lambda s: s.dropna().astype(str).nunique())
    .reset_index(name='distinct_lessons_per_date')
)
user_repeat_flag = (user_repeat[user_repeat['distinct_lessons_per_date'] > 10]
                    .groupby('user_id')['date_parsed'].nunique()
                    .reset_index(name='repeat_count_dates_gt_10'))

# merge repeat to license_agg per user
license_agg = license_agg.merge(user_repeat_flag, on='user_id', how='left')
license_agg['repeat_count_dates_gt_10'] = license_agg['repeat_count_dates_gt_10'].fillna(0).astype(int)

st.dataframe(license_agg.sort_values(['date_parsed','user_id','videos_hit_license'], ascending=[False,True,False]).head(200), use_container_width=True)

# allow selecting user by clicking selectbox
clicked_user = st.selectbox('Select User to filter all views (or use sidebar)', options=[''] + user_list_sorted)
if clicked_user:
    selected_user = clicked_user

# -------------------------
# Lesson test aggregation logic (SQL -> Python)
# -------------------------
st.header('Lesson / Test / CM submissions (per user-device-day)')
if ldf.empty:
    st.info('No lesson_test_submissions file uploaded')
else:
    # normalize content_sub_type: may be dict-like or string
    def sub_type_is_one(x):
        try:
            # if dict-like string, try to detect 'numberint'
            if isinstance(x, str) and 'numberint' in x:
                return '1' in x
            return str(x).strip() in ['1', "{numberint=1}"]
        except Exception:
            return False

    ldf['is_qbank'] = ((ldf['content_type'] == 'lesson') & ldf['content_sub_type'].apply(lambda x: sub_type_is_one(x)))
    ldf['is_test'] = ((ldf['content_type'] == 'test') & ldf['content_sub_type'].isna())

    # group by date,user,device
    qb_agg = (
        ldf.groupby(['submit_date','user_id','device_id'], dropna=False)
        .agg(
            qbank_submits=('content_id', lambda s: s[ldf.loc[s.index,'is_qbank']].dropna().nunique() if len(s)>0 else 0),
            test_submits=('content_id', lambda s: s[ldf.loc[s.index,'is_test']].dropna().nunique() if len(s)>0 else 0)
        )
        .reset_index()
    )

    # custom modules by joined c table: we simulated by counting custom_module_id if present
    if 'custom_module_id' in ldf.columns:
        cm_agg = ldf.groupby(['submit_date','user_id','device_id'])['custom_module_id'].nunique().reset_index(name='cm_submits')
        qb_agg = qb_agg.merge(cm_agg, on=['submit_date','user_id','device_id'], how='left')
    else:
        qb_agg['cm_submits'] = 0

    st.dataframe(qb_agg.sort_values(['submit_date','user_id'], ascending=[False,True]).head(200), use_container_width=True)

# -------------------------
# Suspicious activity logs table (display)
# -------------------------
st.header('Suspicious activity logs (raw)')
if susp.empty:
    st.info('No suspicious_activity_logs uploaded')
else:
    susp['created_on_dt'] = pd.to_datetime(susp['created_on'], errors='coerce')
    st.dataframe(susp[['created_on_dt','user_id','x_real_ip','message']].head(300), use_container_width=True)

# -------------------------
# Cross-filtering when user selected
# -------------------------
if selected_user:
    st.markdown(f"### Filtered views for user: **{selected_user}**")
    lf = license_agg[license_agg['user_id'].astype(str) == selected_user]
    st.dataframe(lf, use_container_width=True)
    if not ldf.empty:
        qf = qb_agg[qb_agg['user_id'].astype(str) == selected_user]
        st.dataframe(qf, use_container_width=True)
    if not susp.empty:
        st.dataframe(susp[susp['user_id'].astype(str) == selected_user], use_container_width=True)

# -------------------------
# Visuals: timeline, device split, radar risk
# -------------------------
st.header('Visuals')
# timeline of videos_hit per day (aggregated across devices)
agg_user_daily = license_agg.groupby(['date_parsed']).agg(total_videos=('videos_hit_license','sum')).reset_index()
if not agg_user_daily.empty:
    fig = px.bar(agg_user_daily, x='date_parsed', y='total_videos', title='Total videos hit (all users) per day')
    st.plotly_chart(fig, use_container_width=True)

# device distribution pie for filtered license
if not license_agg.empty:
    dev_counts = license_agg.groupby('device_id')['videos_hit_license'].sum().reset_index()
    dev_counts = dev_counts[dev_counts['device_id'].notna()]
    if not dev_counts.empty:
        fig2 = px.pie(dev_counts, names='device_id', values='videos_hit_license', title='Device share (videos)')
        st.plotly_chart(fig2, use_container_width=True)

# Simple risk radar (example heuristic)
st.header('Simple risk score & breakdown')
# create basic user-level risk metrics if a user selected
if selected_user:
    ua = license_df[license_df['user_id'].astype(str)==selected_user]
    if not ua.empty:
        # metrics
        total_videos = ua['lesson_id'].dropna().astype(str).nunique()
        devices = ua['device_id'].nunique()
        days_active = ua['date_parsed'].nunique()
        hours_est = ua.get('playback_minutes', pd.Series()).dropna().sum() / 60 if 'playback_minutes' in ua.columns else 0
        over90 = ua.get('quality', pd.Series()).dropna().shape[0]

        # normalize to 0-1 for radar
        vals = [min(total_videos/200,1), min(devices/5,1), min(days_active/30,1), min(hours_est/50,1), min(over90/100,1)]
        labels = ['Videos','Devices','Days active','Hours','Over90']
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=vals, theta=labels, fill='toself', name=selected_user))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False, title='Risk Radar (sample normalized metrics)')
        st.plotly_chart(radar, use_container_width=True)

# -------------------------
# Downloads
# -------------------------
st.header('Downloads')
st.download_button('Download license agg CSV', license_agg.to_csv(index=False).encode('utf-8'), 'license_agg.csv')
if not ldf.empty:
    st.download_button('Download qbank/test agg CSV', qb_agg.to_csv(index=False).encode('utf-8'), 'qbank_agg.csv')

st.markdown('---')
st.caption('This is a starter one-page Streamlit app. You can expand the visuals (timeline per device, geo maps, cohort comparisons, PDF report export) as next steps.')
