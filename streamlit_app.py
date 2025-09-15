import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="User Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 User Analytics Dashboard")
st.markdown("---")

# File upload section
st.sidebar.header("Upload Data Files")
video_license_file = st.sidebar.file_uploader("Video License Logs CSV", type=['csv'])
lesson_test_file = st.sidebar.file_uploader("Lesson Test Submissions CSV", type=['csv'])
video_meta_file = st.sidebar.file_uploader("Video Meta CSV", type=['csv'])
suspicious_activity_file = st.sidebar.file_uploader("Suspicious Activity Logs CSV", type=['csv'])

# Load data function
@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

# Load all datasets
video_license_df = load_data(video_license_file)
lesson_test_df = load_data(lesson_test_file)
video_meta_df = load_data(video_meta_file)
suspicious_activity_df = load_data(suspicious_activity_file)

# Check if all files are uploaded
if all(df is not None for df in [video_license_df, lesson_test_df, video_meta_df, suspicious_activity_df]):
    
    # Convert date columns to datetime
    video_license_df['date'] = pd.to_datetime(video_license_df['date'])
    lesson_test_df['submitted_on'] = pd.to_datetime(lesson_test_df['submitted_on'], unit='ms')
    suspicious_activity_df['date'] = pd.to_datetime(suspicious_activity_df['date'])
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📹 Video License Analysis", 
        "📝 Test Submissions Analysis", 
        "🎥 Video Meta Analysis", 
        "🚨 Suspicious Activity"
    ])
    
    with tab1:
        st.header("Video License Analysis")
        
        # License date, User id, Device ID analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("License Date Range")
            min_date = video_license_df['date'].min()
            max_date = video_license_df['date'].max()
            st.write(f"**From:** {min_date.strftime('%Y-%m-%d')}")
            st.write(f"**To:** {max_date.strftime('%Y-%m-%d')}")
        
        with col2:
            st.subheader("Unique Users & Devices")
            unique_users = video_license_df['user_id'].nunique()
            unique_devices = video_license_df['device_id'].nunique()
            st.write(f"**Unique Users:** {unique_users}")
            st.write(f"**Unique Devices:** {unique_devices}")
        
        with col3:
            st.subheader("Total License Hits")
            total_licenses = len(video_license_df)
            st.write(f"**Total License Requests:** {total_licenses}")
        
        # No of videos hit license per device
        st.subheader("Videos per Device Analysis")
        videos_per_device = video_license_df.groupby(['device_id', 'user_id']).agg({
            'video_id': 'nunique',
            'date': 'nunique'
        }).reset_index()
        videos_per_device.columns = ['device_id', 'user_id', 'unique_videos', 'unique_days']
        
        fig1 = px.histogram(videos_per_device, x='unique_videos', 
                           title='Distribution of Unique Videos per Device')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Repeat Count license (count distinct of date where license count > 10 for that user)
        st.subheader("High-Frequency Users (License Count > 10 per day)")
        daily_license_count = video_license_df.groupby(['user_id', 'date']).size().reset_index(name='license_count')
        high_freq_users = daily_license_count[daily_license_count['license_count'] > 10]
        
        if not high_freq_users.empty:
            repeat_counts = high_freq_users.groupby('user_id')['date'].nunique().reset_index()
            repeat_counts.columns = ['user_id', 'high_frequency_days']
            
            fig2 = px.bar(repeat_counts, x='user_id', y='high_frequency_days',
                         title='Users with High License Frequency (>10/day)')
            st.plotly_chart(fig2, use_container_width=True)
            
            st.dataframe(repeat_counts.sort_values('high_frequency_days', ascending=False))
        else:
            st.info("No users with license count > 10 per day found.")
        
        # Subjects accessed - per device
        st.subheader("Subjects Accessed per Device")
        if 'course_id' in video_license_df.columns and 'device_id' in video_license_df.columns:
            subjects_per_device = video_license_df.groupby(['device_id', 'user_id'])['course_id'].nunique().reset_index()
            subjects_per_device.columns = ['device_id', 'user_id', 'unique_subjects']
            
            fig3 = px.box(subjects_per_device, y='unique_subjects', 
                         title='Distribution of Unique Subjects per Device')
            st.plotly_chart(fig3, use_container_width=True)
        
        # Display raw data
        st.subheader("Raw Video License Data")
        st.dataframe(video_license_df.head(100))
    
    with tab2:
        st.header("Lesson Test Submissions Analysis")
        
        # Convert the SQL logic to Python
        def analyze_test_submissions(lesson_df):
            # Create a copy to avoid modifying original data
            df = lesson_df.copy()
            
            # Ensure submitted_on is datetime
            df['submit_date'] = pd.to_datetime(df['submitted_on']).dt.date
            
            # Qbank submits (content_type = 'lesson' AND content_sub_type = '1')
            qbank_mask = (df['content_type'] == 'lesson') & (df['content_sub_type'] == '1')
            qbank_submits = df[qbank_mask].groupby(['user_id', 'device_id', 'submit_date'])['content_id'].nunique().reset_index()
            qbank_submits.columns = ['user_id', 'device_id', 'submit_date', 'qbank_submits']
            
            # Test submits (content_type = 'test' AND content_sub_type IS NULL)
            test_mask = (df['content_type'] == 'test') & (df['content_sub_type'].isna())
            test_submits = df[test_mask].groupby(['user_id', 'device_id', 'submit_date'])['content_id'].nunique().reset_index()
            test_submits.columns = ['user_id', 'device_id', 'submit_date', 'test_submits']
            
            # Merge results
            result = pd.merge(qbank_submits, test_submits, 
                            on=['user_id', 'device_id', 'submit_date'], 
                            how='outer').fillna(0)
            
            # For custom module submits, we'd need the custom_module_answer table
            # Since we don't have it, we'll note this limitation
            result['cm_submits'] = 0  # Placeholder
            
            return result
        
        test_analysis_df = analyze_test_submissions(lesson_test_df)
        
        # Display analysis
        st.subheader("Daily Test Submission Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_qbank = test_analysis_df['qbank_submits'].sum()
            st.metric("Total QBank Submits", total_qbank)
        
        with col2:
            total_test = test_analysis_df['test_submits'].sum()
            st.metric("Total Test Submits", total_test)
        
        with col3:
            st.metric("Total Custom Module Submits", "N/A")
        
        # Time series of submissions
        daily_submissions = test_analysis_df.groupby('submit_date').agg({
            'qbank_submits': 'sum',
            'test_submits': 'sum'
        }).reset_index()
        
        fig = px.line(daily_submissions, x='submit_date', y=['qbank_submits', 'test_submits'],
                     title='Daily Submission Trends', labels={'value': 'Count', 'variable': 'Type'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Top users by submissions
        top_users = test_analysis_df.groupby('user_id').agg({
            'qbank_submits': 'sum',
            'test_submits': 'sum'
        }).sum(axis=1).sort_values(ascending=False).head(10)
        
        fig2 = px.bar(x=top_users.index, y=top_users.values,
                     title='Top 10 Users by Total Submissions')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display raw data
        st.subheader("Raw Test Submission Data")
        st.dataframe(lesson_test_df.head(100))
    
    with tab3:
        st.header("Video Meta Analysis")
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            unique_lessons = video_meta_df['lesson_id'].nunique()
            st.metric("Unique Lessons", unique_lessons)
        
        with col2:
            unique_subjects = video_meta_df['_subject_title'].nunique()
            st.metric("Unique Subjects", unique_subjects)
        
        with col3:
            avg_duration = video_meta_df['_duration'].mean()
            st.metric("Average Duration (min)", f"{avg_duration/60:.1f}")
        
        # Subject distribution
        subject_dist = video_meta_df['_subject_title'].value_counts().reset_index()
        subject_dist.columns = ['Subject', 'Count']
        
        fig = px.pie(subject_dist, values='Count', names='Subject',
                    title='Video Distribution by Subject')
        st.plotly_chart(fig, use_container_width=True)
        
        # Duration distribution
        fig2 = px.histogram(video_meta_df, x='_duration',
                           title='Video Duration Distribution')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display raw data
        st.subheader("Raw Video Meta Data")
        st.dataframe(video_meta_df.head(100))
    
    with tab4:
        st.header("Suspicious Activity Analysis")
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            total_alerts = len(suspicious_activity_df)
            st.metric("Total Alerts", total_alerts)
        
        with col2:
            unique_users_alerted = suspicious_activity_df['user_id'].nunique()
            st.metric("Users with Alerts", unique_users_alerted)
        
        with col3:
            high_risk_alerts = len(suspicious_activity_df[suspicious_activity_df['alert_level'] == 'high'])
            st.metric("High Risk Alerts", high_risk_alerts)
        
        # Alert level distribution
        alert_dist = suspicious_activity_df['alert_level'].value_counts().reset_index()
        alert_dist.columns = ['Alert Level', 'Count']
        
        fig = px.pie(alert_dist, values='Count', names='Alert Level',
                    title='Alert Level Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        category_dist = suspicious_activity_df['category'].value_counts().reset_index()
        category_dist.columns = ['Category', 'Count']
        
        fig2 = px.bar(category_dist, x='Category', y='Count',
                     title='Suspicious Activity by Category')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Time series of alerts
        alerts_over_time = suspicious_activity_df.groupby('date').size().reset_index(name='count')
        fig3 = px.line(alerts_over_time, x='date', y='count',
                      title='Suspicious Activity Over Time')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Display raw data
        st.subheader("Raw Suspicious Activity Data")
        st.dataframe(suspicious_activity_df.head(100))

else:
    st.warning("Please upload all four CSV files to begin analysis.")
    st.info("""
    Required files:
    1. Video License Logs CSV
    2. Lesson Test Submissions CSV
    3. Video Meta CSV
    4. Suspicious Activity Logs CSV
    """)
