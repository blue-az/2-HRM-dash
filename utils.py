"""Unified utility functions for the fitness dashboard."""

import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

def format_duration(minutes):
    """Format duration in minutes to HH:MM:SS format."""
    if pd.isna(minutes):
        return "N/A"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes * 60) % 60)
    
    return f"{hours:02d}:{mins:02d}:{secs:02d}"

def convert_duration_to_minutes(duration_str):
    """Convert duration string (HH:MM:SS) to minutes."""
    try:
        if pd.isna(duration_str):
            return None
        time = datetime.strptime(str(duration_str), '%H:%M:%S')
        return time.hour * 60 + time.minute + time.second / 60
    except Exception:
        try:
            # Try to convert directly from milliseconds to minutes
            return float(duration_str) / 60000
        except:
            return None

def detect_activity_type(activity_name, default="Other"):
    """Detect activity type from activity name."""
    activity_name = str(activity_name).lower()
    
    activity_keywords = {
        'run': 'Running', 'jog': 'Running',
        'walk': 'Walking', 'hike': 'Walking', 'trek': 'Walking',
        'cycle': 'Outdoor Cycling', 'bike': 'Outdoor Cycling', 
        'mountain bike': 'Outdoor Cycling',
        'spinning': 'Indoor Cycling', 'indoor cycling': 'Indoor Cycling',
        'swim': 'Swimming', 'pool': 'Swimming',
        'tennis': 'Tennis', 'elliptical': 'Elliptical', 'yoga': 'Yoga',
        'strength': 'Strength Training',
        'workout': 'Gym Workout', 'gym': 'Gym Workout',
        'cardio': 'Cardio'
    }
    
    for keyword, activity_type in activity_keywords.items():
        if keyword in activity_name:
            return activity_type
    
    return default

def get_date_range_options(df, key_prefix=None):
    """Generate common date range options for dashboards."""
    today = datetime.now().date()
    
    options = {
        "Last 7 Days": (today - timedelta(days=7), today),
        "Last 30 Days": (today - timedelta(days=30), today),
        "Last 90 Days": (today - timedelta(days=90), today),
        "This Year": (datetime(today.year, 1, 1).date(), today),
        "Last Year": (datetime(today.year-1, 1, 1).date(), datetime(today.year-1, 12, 31).date()),
        "All Time": (df['date'].min() if not df.empty and 'date' in df.columns else None, 
                    df['date'].max() if not df.empty and 'date' in df.columns else None)
    }
    
    # Create a radio button for date range selection
    radio_key = f"{key_prefix}_date_range" if key_prefix else None
    selected_option = st.radio(
        "Select Date Range",
        options=list(options.keys()),
        horizontal=True,
        key=radio_key
    )
    
    return options[selected_option]

def calculate_summary_stats(df, numeric_cols=None, key=None):
    """Calculate summary statistics for dashboard display."""
    if df.empty:
        return pd.DataFrame()
    
    if numeric_cols is None:
        numeric_cols = ['avg_hr', 'max_hr', 'calories', 'duration']
    
    # Filter to only include columns that exist in the dataframe
    valid_cols = [col for col in numeric_cols if col in df.columns]
    
    if not valid_cols:
        return pd.DataFrame()
    
    # Calculate statistics
    stats = df[valid_cols].agg(['mean', 'std', 'min', 'max']).round(2)
    stats.index = ['Average', 'Std Dev', 'Minimum', 'Maximum']
    
    # Display the stats
    if key:
        st.dataframe(stats, use_container_width=True, key=key)
    
    return stats

def generate_color_palette(n_colors, base_palette=None):
    """Generate a color palette for consistent visualization."""
    if base_palette is None:
        # Default color palette
        base_palette = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]
    
    # If we need more colors than in the base palette, we'll cycle through
    return [base_palette[i % len(base_palette)] for i in range(n_colors)]

def create_download_button(df, filename="fitness_data.csv", key=None):
    """Create a download button for the dataframe."""
    if df.empty:
        return
    
    # Convert dataframe to CSV
    csv = df.to_csv(index=False).encode('utf-8')
    
    # Add download button
    button_key = f"{key}_download" if key else None
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=button_key
    )
