"""Visualization modules for fitness data dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config

def plot_time_series(df, metrics, title="Metrics Over Time"):
    """Create a time series line chart for selected metrics."""
    if df.empty or not metrics:
        st.warning("No data available for the selected filters.")
        return
    
    fig = go.Figure()
    for metric in metrics:
        for source in df['source'].unique():
            source_data = df[df['source'] == source]
            if not source_data.empty:
                fig.add_trace(go.Scatter(
                    x=source_data['date'], y=source_data[metric],
                    mode='lines+markers', name=f"{source} - {metric}",
                    line=dict(width=2, shape='spline')
                ))
    
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Value",
        height=config.PLOT_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_activity_distribution(df, title="Activity Type Distribution"):
    """Create a bar chart showing activity type distribution by source."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Group by source and activity type
    activity_counts = df.groupby(['source', 'activity_type']).size().reset_index(name='count')
    
    # Create bar chart
    fig = px.bar(
        activity_counts, x='activity_type', y='count', color='source',
        barmode='group', title=title
    )
    
    fig.update_layout(
        xaxis_title="Activity Type", yaxis_title="Count",
        height=config.PLOT_HEIGHT, margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder':'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_matrix(df, dimensions, color_by="source", title="Relationships Between Metrics"):
    """Create a scatter matrix plot for exploring relationships between metrics."""
    if df.empty or not dimensions:
        st.warning("No data available for the selected filters.")
        return
    
    # Validate dimensions exist in the dataframe
    valid_dimensions = [dim for dim in dimensions if dim in df.columns]
    if not valid_dimensions:
        st.warning("No valid dimensions selected for scatter matrix.")
        return
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        df, dimensions=valid_dimensions, color=color_by,
        title=title, opacity=0.7
    )
    
    fig.update_layout(
        height=config.PLOT_HEIGHT, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Remove upper triangle and diagonal
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_duration_distribution(df, title="Activity Duration Distribution"):
    """Create a box plot showing activity duration distribution."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Group by option
    groupby = st.radio(
        "Group duration by:",
        options=["Source", "Activity Type", "Both"],
        horizontal=True
    )
    
    # Set x and color columns based on grouping option
    x_column = 'activity_type' if groupby in ["Activity Type", "Both"] else 'source'
    color_column = 'source' if groupby in ["Source", "Both"] else 'activity_type'
    
    # Create box plot
    fig = px.box(
        df, x=x_column, y='duration', color=color_column,
        points="all", title=title
    )
    
    fig.update_layout(
        xaxis_title=x_column.replace('_', ' ').title(),
        yaxis_title="Duration (minutes)",
        height=config.PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder':'mean descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_heart_rate_zones(df, title="Heart Rate Zone Distribution"):
    """Create heart rate zone distribution visualization."""
    if df.empty or 'avg_hr' not in df.columns or 'max_hr' not in df.columns:
        st.warning("No heart rate data available for the selected filters.")
        return
    
    # Define heart rate zones
    zones = {
        'Zone 1 (Very Light)': (0.5, 0.6),
        'Zone 2 (Light)': (0.6, 0.7),
        'Zone 3 (Moderate)': (0.7, 0.8),
        'Zone 4 (Hard)': (0.8, 0.9),
        'Zone 5 (Maximum)': (0.9, 1.0)
    }
    
    # Calculate time in each zone
    zone_data = []
    for _, row in df.iterrows():
        if pd.isna(row['avg_hr']) or pd.isna(row['max_hr']) or pd.isna(row['duration']):
            continue
            
        hr_pct = row['avg_hr'] / row['max_hr']
        for zone_name, (min_pct, max_pct) in zones.items():
            if min_pct <= hr_pct < max_pct:
                zone_data.append({
                    'source': row['source'],
                    'activity_type': row['activity_type'],
                    'zone': zone_name,
                    'duration': row['duration']
                })
                break
    
    if not zone_data:
        st.warning("Not enough heart rate data to calculate zones.")
        return
    
    # Create sunburst chart
    zone_df = pd.DataFrame(zone_data)
    fig = px.sunburst(
        zone_df, path=['source', 'zone', 'activity_type'],
        values='duration', title=title
    )
    
    fig.update_layout(
        height=config.PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_bike_analysis(df, title="Bike/Cycling Analysis"):
    """Create visualizations for cycling activities."""
    # Filter to cycling activities
    bike_df = df[df['activity_type'].isin(['Outdoor Cycling', 'Indoor Cycling', 'Biking'])]
    
    if bike_df.empty:
        st.warning("No cycling activities found in the selected data.")
        return
    
    # Cycling metrics over time
    st.subheader("Cycling Metrics Over Time")
    metrics = ['avg_hr', 'max_hr', 'calories', 'duration']
    valid_metrics = [m for m in metrics if m in bike_df.columns]
    
    if valid_metrics:
        metric_option = st.selectbox(
            "Select metric to view:",
            options=valid_metrics,
            format_func=lambda x: {
                'avg_hr': 'Average Heart Rate', 
                'max_hr': 'Maximum Heart Rate',
                'calories': 'Calories Burned',
                'duration': 'Duration (minutes)'
            }.get(x, x.replace('_', ' ').title())
        )
        
        fig = px.line(
            bike_df.sort_values('date'), x='date', y=metric_option,
            color='source', markers=True,
            title=f"{metric_option.replace('_', ' ').title()} for Cycling Activities"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=metric_option.replace('_', ' ').title(),
            height=config.PLOT_HEIGHT
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cycling activity breakdown by location if available
    if 'location' in bike_df.columns and not bike_df['location'].isna().all():
        st.subheader("Cycling by Location")
        
        # Group by location and source
        location_stats = bike_df.groupby(['location', 'source']).agg({
            'duration': ['mean', 'count'],
            'avg_hr': 'mean',
            'calories': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        location_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in location_stats.columns]
        
        # Create bar chart
        fig = px.bar(
            location_stats,
            x='location', y='duration_count',
            color='source',
            hover_data=['duration_mean', 'avg_hr_mean', 'calories_mean'],
            title="Cycling Activities by Location"
        )
        
        fig.update_layout(
            xaxis_title="Location",
            yaxis_title="Number of Activities",
            height=config.PLOT_HEIGHT
        )
        
        st.plotly_chart(fig, use_container_width=True)

def plot_tennis_analysis(df, title="Tennis Analysis"):
    """Create visualizations for tennis activities."""
    # Filter to tennis activities
    tennis_df = df[df['activity_type'] == 'Tennis']
    
    if tennis_df.empty:
        st.warning("No tennis activities found in the selected data.")
        return
    
    # Tennis frequency over time
    st.subheader("Tennis Activity Frequency")
    
    # Group by month
    tennis_df['month'] = pd.to_datetime(tennis_df['date']).dt.to_period('M')
    monthly_counts = tennis_df.groupby(['month', 'source']).size().reset_index(name='count')
    monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()
    
    # Create bar chart
    fig = px.bar(
        monthly_counts,
        x='month', y='count',
        color='source',
        title="Tennis Activity Frequency by Month"
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Sessions",
        height=config.PLOT_HEIGHT
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tennis metrics
    metrics = ['avg_hr', 'duration', 'calories']
    valid_metrics = [m for m in metrics if m in tennis_df.columns]
    
    if valid_metrics:
        selected_metric = st.selectbox(
            "Select metric:",
            options=valid_metrics,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        fig = px.line(
            tennis_df.sort_values('date'),
            x='date', y=selected_metric,
            color='source', markers=True,
            title=f"{selected_metric.replace('_', ' ').title()} During Tennis"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=selected_metric.replace('_', ' ').title(),
            height=config.PLOT_HEIGHT
        )
        
        st.plotly_chart(fig, use_container_width=True)

def plot_multi_source_comparison(df, title="Data Source Comparison"):
    """Create visualizations comparing data across different sources."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Get available sources
    sources = df['source'].unique()
    
    if len(sources) <= 1:
        st.info("At least two data sources are needed for comparison. Please select more data sources in the filters.")
        return
    
    # Create comparison plots for different metrics
    metrics = [
        {'column': 'avg_hr', 'title': 'Average Heart Rate by Activity Type', 'label': 'Average Heart Rate (bpm)'},
        {'column': 'calories', 'title': 'Average Calories by Activity Type', 'label': 'Average Calories'},
        {'column': 'duration', 'title': 'Average Duration by Activity Type', 'label': 'Average Duration (minutes)'}
    ]
    
    for metric in metrics:
        if metric['column'] in df.columns and not df[metric['column']].isna().all():
            # Group by source and activity type
            metric_by_activity = df.groupby(['source', 'activity_type'])[metric['column']].mean().reset_index()
            
            if not metric_by_activity.empty:
                fig = px.bar(
                    metric_by_activity,
                    x='activity_type', y=metric['column'],
                    color='source', barmode='group',
                    title=metric['title']
                )
                
                fig.update_layout(
                    xaxis_title="Activity Type",
                    yaxis_title=metric['label'],
                    height=config.PLOT_HEIGHT
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Activity count comparison
    activity_counts = df.groupby(['source', 'activity_type']).size().reset_index(name='count')
    
    fig = px.bar(
        activity_counts,
        x='activity_type', y='count',
        color='source', barmode='group',
        title="Activity Count by Type and Source"
    )
    
    fig.update_layout(
        xaxis_title="Activity Type",
        yaxis_title="Number of Activities",
        height=config.PLOT_HEIGHT
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_summary_stats(df):
    """Display summary statistics for the filtered data."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Calculate stats by source
    stats = df.groupby('source')[['avg_hr', 'max_hr', 'calories', 'duration']].agg(['mean', 'max']).round(1)
    
    # Reformat column names
    stats.columns = [f"{col[0]} ({col[1]})" for col in stats.columns]
    
    # Display as table
    st.dataframe(stats, use_container_width=True)
    
    # Show date range if available
    if 'date' in df.columns:
        st.text(f"Date Range: {df['date'].min()} to {df['date'].max()}")
