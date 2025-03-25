"""
Fitness Dashboard - Main Application

A Streamlit dashboard for visualizing and analyzing fitness data
from multiple sources (MiiFit, Garmin, PolarF11, and ChargeHR).
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import config
import data_processor as dp
import visualization as viz
from utils import format_duration

# Page configuration
st.set_page_config(
    page_title="Fitness Data Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data(ttl=600)
def load_data():
    """Load and combine data from all sources."""
    return dp.combine_data()

def sidebar_controls(df):
    """Create the sidebar controls."""
    st.sidebar.header("Controls")
    
    # Display current data file locations
    with st.sidebar.expander("Data Source Locations"):
        for source, path in {
            'MiiFit': config.MIIFIT_PATH,
            'Garmin': config.GARMIN_PATH,
            'PolarF11': config.POLAR_F11_PATH,
            'ChargeHR': config.CHARGE_HR_PATH
        }.items():
            st.write(f"{source}: {os.path.abspath(path)}")
            st.write(f"File exists: {os.path.exists(path)}")
    
    # Date range selector
    min_date = df['date'].min() if not df.empty and 'date' in df.columns else datetime(2020, 1, 1).date()
    max_date = df['date'].max() if not df.empty and 'date' in df.columns else datetime.now().date()
    default_start = max_date - timedelta(days=config.DEFAULT_DAYS_BACK)
    
    start_date = st.sidebar.date_input(
        "Start Date", default_start,
        min_value=min_date, max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date", max_date,
        min_value=start_date, max_value=max_date
    )
    
    # Source selector
    available_sources = sorted(df['source'].unique()) if not df.empty else []
    selected_sources = st.sidebar.multiselect(
        "Select Data Sources", options=available_sources,
        default=available_sources[:2] if len(available_sources) >= 2 else available_sources
    )
    
    # Activity type selector
    st.sidebar.markdown("---")
    show_activity_filter = st.sidebar.checkbox("Filter by Activity Type", False)
    
    selected_activities = []
    if show_activity_filter and not df.empty:
        available_activities = sorted(df['activity_type'].unique()) 
        selected_activities = st.sidebar.multiselect(
            "Select Activity Types", options=available_activities
        )
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'sources': selected_sources,
        'activity_types': selected_activities
    }

def show_quick_overview_tab(filtered_df):
    """Display a simplified overview dashboard."""
    st.title("Heart Rate Monitor Dashboard")
    
    # Summary statistics
    st.header("Summary Statistics")
    stats = dp.get_summary_stats(filtered_df)
    st.dataframe(stats, use_container_width=True)
    
    # Heart rate line chart
    st.header("Heart Rate Over Time")
    hr_metrics = st.multiselect(
        "Select HR Metrics", ['avg_hr', 'max_hr'],
        default=['avg_hr', 'max_hr']
    )
    
    if hr_metrics and not filtered_df.empty:
        hr_data = dp.aggregate_data_by_date(filtered_df, hr_metrics)
        viz.plot_time_series(hr_data, hr_metrics, "Heart Rate Over Time")
    
    # Activity duration distribution
    st.header("Activity Duration")
    viz.plot_duration_distribution(filtered_df)
    
    # Scatter matrix for metric relationships
    st.header("Metric Relationships")
    viz.plot_scatter_matrix(filtered_df, ['avg_hr', 'max_hr', 'calories', 'duration'])
    
    # Activity breakdown
    st.header("Activity Type Distribution")
    viz.plot_activity_distribution(filtered_df)

def show_overview_tab(filtered_df):
    """Display content for the detailed Overview tab."""
    st.header("Fitness Dashboard Overview")
    
    # Data sources status
    st.subheader("Data Sources Status")
    data_sources = {source: any(filtered_df['source'] == source) for source in ['MiiFit', 'Garmin', 'PolarF11', 'ChargeHR']}
    
    # Display source status and counts
    col1, col2 = st.columns(2)
    with col1:
        for source, present in data_sources.items():
            st.write(f"- {source}: {'✅ Loaded' if present else '❌ Not loaded'}")
    
    with col2:
        if not filtered_df.empty:
            source_counts = filtered_df['source'].value_counts()
            for source, count in source_counts.items():
                st.metric(f"{source} Activities", count)
    
    # Summary metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Activity Count")
        st.metric("Total Activities", len(filtered_df))
    
    with col2:
        st.subheader("Date Range")
        if 'date' in filtered_df.columns and not filtered_df.empty:
            min_date = filtered_df['date'].min()
            max_date = filtered_df['date'].max()
            date_range = (max_date - min_date).days + 1
            
            st.metric("Start Date", min_date.strftime('%Y-%m-%d'))
            st.metric("End Date", max_date.strftime('%Y-%m-%d'))
            st.metric("Days", date_range)
    
    with col3:
        st.subheader("Average Stats")
        if not filtered_df.empty:
            if 'duration' in filtered_df.columns:
                st.metric("Avg Duration", format_duration(filtered_df['duration'].mean()))
            if 'calories' in filtered_df.columns:
                st.metric("Avg Calories", f"{filtered_df['calories'].mean():.1f}")
            if 'avg_hr' in filtered_df.columns:
                st.metric("Avg Heart Rate", f"{filtered_df['avg_hr'].mean():.1f} bpm")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    viz.display_summary_stats(filtered_df)
    
    # Recent activities
    st.subheader("Recent Activities")
    recent = filtered_df.sort_values('date', ascending=False).head(5)
    
    for _, act in recent.iterrows():
        with st.expander(f"{act['date']} - {act['activity_type']} ({act['source']})"):
            cols = st.columns(4)
            metrics = [
                ("Duration", format_duration(act['duration'])),
                ("Calories", f"{act['calories']:.0f}" if not pd.isna(act['calories']) else "N/A"),
                ("Avg HR", f"{act['avg_hr']:.0f} bpm" if not pd.isna(act['avg_hr']) else "N/A"),
                ("Max HR", f"{act['max_hr']:.0f} bpm" if not pd.isna(act['max_hr']) else "N/A")
            ]
            
            for i, (label, value) in enumerate(metrics):
                with cols[i]:
                    st.metric(label, value)

def show_tabs(filtered_df):
    """Create and populate dashboard tabs."""
    tab_names = [
        "Quick Overview", "Detailed Overview", 
        "Heart Rate Analysis", "Activity Distribution", 
        "Duration Analysis", "Metric Relationships",
        "Bike Analysis", "Tennis Analysis", 
        "Source Comparison", "Raw Data"
    ]
    
    tabs = st.tabs(tab_names)
    
    # Tab 0: Quick Overview
    with tabs[0]:
        show_quick_overview_tab(filtered_df)
    
    # Tab 1: Detailed Overview
    with tabs[1]:
        show_overview_tab(filtered_df)
    
    # Tab 2: Heart Rate Analysis
    with tabs[2]:
        st.header("Heart Rate Analysis")
        hr_metrics = st.multiselect(
            "Select Heart Rate Metrics",
            options=['avg_hr', 'max_hr'],
            default=['avg_hr', 'max_hr']
        )
        
        if hr_metrics:
            hr_data = dp.aggregate_data_by_date(filtered_df, hr_metrics)
            viz.plot_time_series(hr_data, hr_metrics, "Heart Rate Over Time")
        
        st.subheader("Heart Rate Zones")
        viz.plot_heart_rate_zones(filtered_df)
    
    # Tab 3: Activity Distribution
    with tabs[3]:
        st.header("Activity Distribution")
        viz.plot_activity_distribution(filtered_df)
        
        # Monthly activity trends
        st.subheader("Activity Trends")
        monthly_counts = dp.calculate_monthly_activity_counts(filtered_df)
        if not monthly_counts.empty:
            fig = px.line(
                monthly_counts, x='date', y='count',
                color='activity_type', markers=True,
                title="Monthly Activity Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Duration Analysis
    with tabs[4]:
        st.header("Duration Analysis")
        viz.plot_duration_distribution(filtered_df)
        
        # Duration over time
        if not filtered_df.empty:
            st.subheader("Duration Over Time")
            duration_by_date = filtered_df.groupby(['date', 'source'])['duration'].sum().reset_index()
            if not duration_by_date.empty:
                fig = px.line(
                    duration_by_date, x='date', y='duration',
                    color='source', markers=True,
                    title="Total Activity Duration by Date"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Metric Relationships
    with tabs[5]:
        st.header("Metric Relationships")
        dimensions = st.multiselect(
            "Select Metrics to Compare",
            options=['avg_hr', 'max_hr', 'calories', 'duration'],
            default=['avg_hr', 'max_hr', 'calories', 'duration']
        )
        
        color_by = st.radio(
            "Color by:", options=['source', 'activity_type'],
            horizontal=True
        )
        
        viz.plot_scatter_matrix(filtered_df, dimensions, color_by)
        
        # Correlation heatmap
        if not filtered_df.empty and dimensions:
            st.subheader("Correlation Matrix")
            corr_df = filtered_df[dimensions].corr()
            fig = px.imshow(
                corr_df, text_auto=True,
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Bike Analysis
    with tabs[6]:
        st.header("Bike/Cycling Analysis")
        viz.plot_bike_analysis(filtered_df)
    
    # Tab 7: Tennis Analysis
    with tabs[7]:
        st.header("Tennis Analysis")
        viz.plot_tennis_analysis(filtered_df)
    
    # Tab 8: Source Comparison
    with tabs[8]:
        st.header("Data Source Comparison")
        viz.plot_multi_source_comparison(filtered_df)
    
    # Tab 9: Raw Data
    with tabs[9]:
        st.header("Raw Data")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Allow CSV download
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="fitness_data.csv",
                mime="text/csv"
            )

def main():
    """Main function to run the dashboard."""
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Create sidebar controls
    filters = sidebar_controls(df)
    
    # Filter data based on selection
    filtered_df = dp.filter_data(
        df, filters['start_date'], filters['end_date'],
        filters['sources'], filters['activity_types']
    )
    
    # Handle no data case
    if filtered_df.empty:
        st.warning("No data available for the selected filters or no data files found.")
        
        # Display troubleshooting info
        with st.expander("Troubleshooting"):
            st.error("""
            ### Troubleshooting Data Loading Issues
            
            1. **Check File Paths**:
               - Files are searched in these locations:
               - Specified paths in config.py
               - Current directory
               - 'data/' folder in current directory
               - Parent directory
               - 'data/' folder in parent directory
               
            2. **Expected File Types**:
               - MiiFit: SQLite database (.db) or CSV
               - Garmin: JSON file or CSV
               - PolarF11: CSV file
               - ChargeHR: CSV file
            """)
        return
    
    # Show dashboard tabs
    show_tabs(filtered_df)

# Import needed visualization components
import plotly.express as px

# Run the application
if __name__ == "__main__":
    main()
