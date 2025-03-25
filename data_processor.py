"""Core data processing module for fitness tracker data."""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
import traceback
import warnings
from datetime import datetime
import config
from utils import convert_duration_to_minutes

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, 
                      message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated')

def create_empty_df():
    """Create an empty DataFrame with the standard columns."""
    return pd.DataFrame(columns=['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location'])

def standardize_dataframe(df, source_name, column_mapping=None):
    """Standardize a dataframe to the common format."""
    # Add source identifier
    df['source'] = source_name
    
    # Add location if missing
    if 'location' not in df.columns:
        df['location'] = None
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    
    # Standardize activity types
    if 'activity_type' in df.columns:
        df['activity_type'] = df['activity_type'].map(
            lambda x: config.ACTIVITY_TYPE_MAPPING.get(x, x) if isinstance(x, str) else x
        )
    
    # Convert duration string to minutes if needed
    if 'duration_string' in df.columns:
        df['duration'] = df['duration_string'].apply(convert_duration_to_minutes)
    
    # Estimate HR from calories for ChargeHR data
    if source_name == 'ChargeHR' and 'calories' in df.columns and ('avg_hr' not in df.columns or df['avg_hr'].isna().all()):
        df['avg_hr'] = df['calories'] / 4
    
    # Ensure all standard columns exist
    std_cols = ['date', 'activity_type', 'avg_hr', 'max_hr', 'calories', 'duration', 'source', 'location']
    for col in std_cols:
        if col not in df.columns:
            df[col] = None
    
    # Keep only standard columns and drop rows with missing dates
    return df[std_cols].dropna(subset=['date'])

def load_miifit_data():
    """Load MiiFit data from db or csv file."""
    path = config.MIIFIT_PATH
    
    try:
        # SQLite database
        if os.path.exists(path) and path.endswith('.db'):
            print(f"Loading MiiFit database from: {path}")
            conn = sqlite3.connect(path)
            df = pd.read_sql("SELECT _id, DATE, TYPE, TRACKID, ENDTIME, CAL, AVGHR, MAX_HR from TRACKRECORD", conn, index_col="_id")
            
            # Process timestamps and calculate duration
            df['TRACKID'] = pd.to_datetime(df['TRACKID'], unit='s')
            df['ENDTIME'] = pd.to_datetime(df['ENDTIME'], unit='s')
            df['duration_minutes'] = ((df['ENDTIME'] - df['TRACKID']).dt.total_seconds() / 60).round()
            
            # Filter out outliers
            df = df[(df["AVGHR"] > 50) & (df["MAX_HR"] > 50) & (df["duration_minutes"] > 10)]
            
            # Map activity types
            type_map = {16: "Free", 10: "IndCyc", 9: "OutCyc", 12: "Elliptical", 60: "Yoga", 14: "Swim"}
            df['TYPE'] = df['TYPE'].replace(type_map)
            
            # Rename columns to standard format
            df = df.rename(columns=config.MIIFIT_COLUMNS)
            
            # Standardize the dataframe
            df = standardize_dataframe(df, 'MiiFit')
            print(f"Processed MiiFit data: {len(df)} rows")
            return df
            
        # CSV file
        csv_path = path if path.endswith('.csv') else path.replace('.db', '.csv')
        if os.path.exists(csv_path):
            print(f"Loading MiiFit CSV from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Map columns to standard names (case-insensitive)
            column_map = {col: std for col in df.columns 
                         for std in config.MIIFIT_COLUMNS 
                         if col.lower() == std.lower()}
            if column_map:
                df = df.rename(columns=column_map)
            
            # Apply standard column mapping
            valid_cols = {k: v for k, v in config.MIIFIT_COLUMNS.items() if k in df.columns}
            df = df.rename(columns=valid_cols)
            
            # Standardize the dataframe
            df = standardize_dataframe(df, 'MiiFit')
            print(f"Processed MiiFit CSV data: {len(df)} rows")
            return df
        
        print(f"No MiiFit data found at: {path}")
        return create_empty_df()
        
    except Exception as e:
        print(f"Error loading MiiFit data: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_garmin_data():
    """Load Garmin data from json or csv file."""
    path = config.GARMIN_PATH
    
    try:
        # JSON file
        if os.path.exists(path) and path.endswith('.json'):
            print(f"Loading Garmin JSON from: {path}")
            with open(path, 'r') as file:
                json_data = json.load(file)
            
            # Extract activities
            activities = json_data[0].get('summarizedActivitiesExport', []) if (
                isinstance(json_data, list) and len(json_data) > 0) else []
            
            if not activities:
                print("No activities found in Garmin JSON")
                return create_empty_df()
            
            # Extract necessary fields
            records = []
            for act in activities:
                start_time = act.get('startTimeLocal')
                if not start_time:
                    continue
                
                records.append({
                    "date": datetime.fromtimestamp(start_time / 1000).date(),
                    "activity_type": act.get('activityType', 'Unknown'),
                    "avg_hr": act.get('avgHr'),
                    "max_hr": act.get('maxHr'),
                    "calories": act.get('calories'),
                    "duration": act.get('duration', 0) / 60000 if act.get('duration', 0) else None,
                    "source": "Garmin",
                    "location": None
                })
            
            # Create and standardize DataFrame
            df = pd.DataFrame(records)
            df = standardize_dataframe(df, 'Garmin')
            print(f"Processed Garmin JSON data: {len(df)} rows")
            return df
        
        # CSV file
        csv_path = path if path.endswith('.csv') else path.replace('.json', '.csv')
        if os.path.exists(csv_path):
            print(f"Loading Garmin CSV from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Map columns to standard names (case-insensitive)
            column_map = {col: std for col in df.columns 
                         for std in config.GARMIN_COLUMNS 
                         if col.lower() == std.lower()}
            if column_map:
                df = df.rename(columns=column_map)
            
            # Apply standard column mapping
            valid_cols = {k: v for k, v in config.GARMIN_COLUMNS.items() if k in df.columns}
            df = df.rename(columns=valid_cols)
            
            # Convert calories to numeric
            if 'calories' in df.columns:
                df['calories'] = pd.to_numeric(df['calories'], errors='coerce')
            
            # Standardize the dataframe
            df = standardize_dataframe(df, 'Garmin')
            print(f"Processed Garmin CSV data: {len(df)} rows")
            return df
        
        print(f"No Garmin data found at: {path}")
        return create_empty_df()
        
    except Exception as e:
        print(f"Error loading Garmin data: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_polar_f11_data():
    """Load Polar F11 data from CSV file."""
    path = config.POLAR_F11_PATH
    
    try:
        if not os.path.exists(path):
            print(f"No Polar F11 data found at: {path}")
            return create_empty_df()
        
        print(f"Loading Polar F11 data from: {path}")
        df = pd.read_csv(path)
        
        # Rename columns to standard format
        df = df.rename(columns=config.POLAR_F11_COLUMNS)
        
        # Process dates (clean .0 suffix)
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        # Convert numeric columns
        for col in ['avg_hr', 'max_hr', 'calories']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize the dataframe
        df = standardize_dataframe(df, 'PolarF11')
        print(f"Processed Polar F11 data: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error loading Polar F11 data: {e}")
        traceback.print_exc()
        return create_empty_df()

def load_charge_hr_data():
    """Load Charge HR data from CSV file."""
    path = config.CHARGE_HR_PATH
    
    try:
        if not os.path.exists(path):
            print(f"No Charge HR data found at: {path}")
            return create_empty_df()
        
        print(f"Loading Charge HR data from: {path}")
        df = pd.read_csv(path)
        
        # Rename columns to standard format
        df = df.rename(columns=config.CHARGE_HR_COLUMNS)
        
        # Process dates (add year if available)
        if 'date' in df.columns and 'year' in df.columns:
            if not df['date'].astype(str).str.contains(r'\d{4}').any():
                df['date'] = df['date'].astype(str) + " " + df['year'].astype(str)
        
        # Standardize the dataframe (this will handle the HR estimation)
        df = standardize_dataframe(df, 'ChargeHR')
        print(f"Processed Charge HR data: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error loading Charge HR data: {e}")
        traceback.print_exc()
        return create_empty_df()

def prepare_df_for_concat(df, columns):
    """Prepare a dataframe for concatenation with proper types."""
    if df.empty:
        return None
        
    # Add missing columns with appropriate types
    for col in columns:
        if col not in df.columns:
            if col in ['avg_hr', 'max_hr', 'calories', 'duration']:
                df[col] = pd.Series(dtype='float64')
            elif col == 'date':
                df[col] = pd.Series(dtype='datetime64[ns]')
            else:
                df[col] = pd.Series(dtype='object')
    return df

def combine_data():
    """Combine data from all sources."""
    print("\n===== LOADING DATA FROM ALL SOURCES =====")
    
    # Load data from all sources
    dataframes = {
        'MiiFit': load_miifit_data(),
        'Garmin': load_garmin_data(),
        'PolarF11': load_polar_f11_data(),
        'ChargeHR': load_charge_hr_data()
    }
    
    # Log source statistics
    for source, df in dataframes.items():
        print(f"{source} data: {len(df)} rows")
    
    # Filter non-empty dataframes
    non_empty_dfs = [df for df in dataframes.values() if not df.empty]
    if not non_empty_dfs:
        print("No data found from any source")
        return create_empty_df()
    
    # Get all columns for type alignment
    all_columns = set()
    for df in non_empty_dfs:
        all_columns.update(df.columns)
    
    # Prepare dataframes for concatenation
    processed_dfs = [prepare_df_for_concat(df, all_columns) for df in non_empty_dfs if not df.empty]
    processed_dfs = [df for df in processed_dfs if df is not None]
    
    # Concatenate dataframes
    if not processed_dfs:
        return create_empty_df()
        
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Post-process combined data
    if not combined_df.empty:
        # Sort by date
        combined_df = combined_df.sort_values('date')
        
        # Convert numeric columns
        for col in ['avg_hr', 'max_hr', 'calories', 'duration']:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    print(f"Combined data: {len(combined_df)} rows")
    return combined_df

def filter_data(df, start_date=None, end_date=None, sources=None, activity_types=None):
    """Filter data based on selected criteria."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply filters in sequence
    if start_date is not None and 'date' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['date'] >= start_date]
    
    if end_date is not None and 'date' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['date'] <= end_date]
    
    if sources and 'source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    if activity_types and 'activity_type' in filtered_df.columns and activity_types:
        filtered_df = filtered_df[filtered_df['activity_type'].isin(activity_types)]
    
    return filtered_df

def aggregate_data_by_date(df, metrics=None):
    """Aggregate data by date for the selected metrics."""
    if df.empty or not metrics or 'date' not in df.columns:
        return pd.DataFrame()
    
    # Use only metrics that exist in the dataframe
    valid_metrics = [metric for metric in metrics if metric in df.columns]
    if not valid_metrics:
        return pd.DataFrame()
    
    # Group by date and source, calculate mean for each metric
    agg_dict = {metric: 'mean' for metric in valid_metrics}
    return df.groupby(['date', 'source']).agg(agg_dict).reset_index()

def calculate_monthly_activity_counts(df):
    """Calculate monthly activity counts by type."""
    if df.empty or 'date' not in df.columns or 'activity_type' not in df.columns:
        return pd.DataFrame()
    
    # Ensure date is datetime
    monthly_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(monthly_df['date']):
        monthly_df['date'] = pd.to_datetime(monthly_df['date'])
    
    # Extract year and month, then group
    monthly_df['year'] = monthly_df['date'].dt.year
    monthly_df['month'] = monthly_df['date'].dt.month
    monthly_counts = monthly_df.groupby(['year', 'month', 'activity_type']).size().reset_index(name='count')
    
    # Create date column for easier plotting
    monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
    return monthly_counts

def get_summary_stats(df):
    """Calculate summary statistics for numeric columns."""
    if df.empty:
        return pd.DataFrame()
    
    # Calculate stats for numeric columns
    numeric_cols = ['avg_hr', 'max_hr', 'calories', 'duration']
    valid_cols = [col for col in numeric_cols if col in df.columns]
    
    if not valid_cols:
        return pd.DataFrame()
    
    # Calculate stats and rename index
    stats = df[valid_cols].agg(['mean', 'std', 'min', 'max']).round(2)
    stats.index = ['Average', 'Std Dev', 'Minimum', 'Maximum']
    return stats

# Create a module for specialized analysis functions
class ActivityAnalysis:
    """Specialized analysis functions for specific activity types."""
    
    @staticmethod
    def calculate_bike_metrics(df):
        """Calculate additional metrics for cycling activities."""
        if df.empty:
            return df
            
        # Filter to cycling activities
        bike_df = df[df['activity_type'].isin(['Outdoor Cycling', 'Indoor Cycling', 'Biking'])]
        if bike_df.empty:
            return bike_df
            
        # Calculate speed if distance is available
        if 'distance' in bike_df.columns and 'duration' in bike_df.columns:
            bike_df.loc[:, 'speed'] = bike_df['distance'] / (bike_df['duration'] / 60)  # mph
            
        # Calculate calories per hour
        if 'calories' in bike_df.columns and 'duration' in bike_df.columns:
            bike_df.loc[:, 'calories_per_hour'] = bike_df['calories'] / (bike_df['duration'] / 60)
            
        return bike_df
    
    @staticmethod
    def calculate_heart_rate_zones(df):
        """Calculate time spent in different heart rate zones."""
        if df.empty or 'avg_hr' not in df.columns or 'max_hr' not in df.columns:
            return pd.DataFrame()
            
        # Define heart rate zones (as % of max HR)
        zones = {
            'Zone 1 (Recovery)': (0.5, 0.6),
            'Zone 2 (Endurance)': (0.6, 0.7),
            'Zone 3 (Tempo)': (0.7, 0.8),
            'Zone 4 (Threshold)': (0.8, 0.9),
            'Zone 5 (Maximum)': (0.9, 1.0)
        }
        
        # Calculate time in each zone for each activity
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
                        'date': row['date'],
                        'zone': zone_name,
                        'duration': row['duration']
                    })
                    break
        
        return pd.DataFrame(zone_data) if zone_data else pd.DataFrame()
    
    @staticmethod
    def calculate_activity_trends(df):
        """Calculate trends in activity metrics over time."""
        if df.empty or 'date' not in df.columns:
            return pd.DataFrame()
            
        # Group by week for trend analysis
        weekly_df = df.copy()
        weekly_df['week'] = pd.to_datetime(weekly_df['date']).dt.to_period('W')
        
        # Aggregate metrics by week
        agg_dict = {}
        for col in ['duration', 'calories', 'avg_hr', 'max_hr']:
            if col in weekly_df.columns:
                agg_dict[col] = ['mean', 'sum']
                
        if not agg_dict:
            return pd.DataFrame()
            
        # Group and calculate stats
        trends = weekly_df.groupby(['week', 'source', 'activity_type']).agg(agg_dict).reset_index()
        trends['week'] = trends['week'].dt.to_timestamp()
        
        # Flatten multi-level columns
        trends.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in trends.columns]
        
        return trends
