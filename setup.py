"""
Setup script for the fitness dashboard.
This script can generate sample data files for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse

def generate_miifit_sample(filename, num_records=50):
    """Generate sample MiiFit data file."""
    print(f"Generating sample MiiFit data with {num_records} records...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates (last 180 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    dates = [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(num_records)]
    dates.sort()
    
    # Activity types
    activity_types = [16, 10, 9, 12, 60, 14]  # Codes for different activities
    
    # Generate data
    data = []
    for i in range(num_records):
        activity_type = np.random.choice(activity_types)
        
        # Calculate duration (20-90 minutes)
        duration = np.random.randint(20, 91)
        
        # Generate start and end timestamps
        start_time = datetime.combine(dates[i], datetime.min.time()) + timedelta(hours=np.random.randint(6, 20))
        end_time = start_time + timedelta(minutes=duration)
        
        # Convert to Unix timestamps
        trackid = int(start_time.timestamp())
        endtime = int(end_time.timestamp())
        
        # Heart rate values
        avg_hr = np.random.randint(90, 160)
        max_hr = avg_hr + np.random.randint(10, 40)
        
        # Calories burned (roughly based on duration and intensity)
        calories = int(duration * avg_hr * 0.05)
        
        data.append({
            "_id": i + 1,
            "DATE": dates[i].strftime("%Y-%m-%d"),
            "TYPE": activity_type,
            "TRACKID": trackid,
            "ENDTIME": endtime,
            "CAL": calories,
            "AVGHR": avg_hr,
            "MAX_HR": max_hr,
            "duration_minutes": duration
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Sample MiiFit data saved to {filename}")
    print(f"Column names: {df.columns.tolist()}")
    return df

def generate_garmin_sample(filename, num_records=50):
    """Generate sample Garmin data file."""
    print(f"Generating sample Garmin data with {num_records} records...")
    
    # Set random seed for reproducibility
    np.random.seed(24)
    
    # Generate dates (last 180 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    dates = [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(num_records)]
    dates.sort()
    
    # Activity types
    activity_types = ["Running", "Walking", "Cycling", "Swimming", "Yoga", "Strength Training", "Indoor Cycling", "Elliptical"]
    
    # Generate data
    data = []
    for i in range(num_records):
        activity_type = np.random.choice(activity_types)
        
        # Calculate duration (20-90 minutes)
        duration_minutes = np.random.randint(20, 91)
        hours = duration_minutes // 60
        minutes = duration_minutes % 60
        seconds = np.random.randint(0, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Heart rate values
        avg_hr = np.random.randint(90, 160)
        max_hr = avg_hr + np.random.randint(10, 40)
        
        # Calories burned (roughly based on duration and intensity)
        calories = int(duration_minutes * avg_hr * 0.05)
        
        # Speed values (if applicable)
        avg_speed = f"{np.random.uniform(4, 12):.1f}"
        max_speed = f"{float(avg_speed) + np.random.uniform(2, 5):.1f}"
        
        data.append({
            "Activity Type": activity_type,
            "Date": dates[i].strftime("%Y-%m-%d"),
            "Favorite": np.random.choice([True, False]),
            "Title": f"{activity_type} on {dates[i].strftime('%b %d')}",
            "Distance": np.random.uniform(1, 15),
            "Calories": str(calories),
            "Total Time": duration_str,
            "Avg HR": avg_hr,
            "Max HR": max_hr,
            "Avg Bike Cadence": f"{np.random.randint(60, 100)}" if "Cycling" in activity_type else "",
            "Max Bike Cadence": f"{np.random.randint(80, 120)}" if "Cycling" in activity_type else "",
            "Avg Speed": avg_speed,
            "Max Speed": max_speed,
            "Avg Stride Length": f"{np.random.uniform(0.8, 1.5):.2f}" if "Running" in activity_type or "Walking" in activity_type else "",
            "Training Stress Score®": np.random.uniform(20, 100) if np.random.random() > 0.5 else None,
            "Decompression": "",
            "Best Lap Time": f"00:0{np.random.randint(1, 6)}:{np.random.randint(10, 60)}" if np.random.random() > 0.7 else "",
            "Number of Laps": np.random.randint(1, 10) if np.random.random() > 0.7 else None,
            "Moving Time": duration_str,
            "Elapsed Time": duration_str
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Sample Garmin data saved to {filename}")
    print(f"Column names: {df.columns.tolist()}")
    return df

def check_configuration():
    """Check the configuration and create sample data if requested."""
    parser = argparse.ArgumentParser(description='Setup the fitness dashboard.')
    parser.add_argument('--generate-sample', action='store_true', help='Generate sample data files')
    parser.add_argument('--miifit-records', type=int, default=50, help='Number of MiiFit records to generate')
    parser.add_argument('--garmin-records', type=int, default=50, help='Number of Garmin records to generate')
    
    args = parser.parse_args()
    
    # Check for configuration file
    if not os.path.exists('config.py'):
        print("Error: config.py not found. Please make sure you're running this script from the correct directory.")
        return False
    
    # Generate sample data if requested
    if args.generate_sample:
        generate_miifit_sample('DataSnippet.csv', args.miifit_records)
        generate_garmin_sample('Activities.csv', args.garmin_records)
        
        # Update the config.py file to use the sample data files
        update_config_for_sample_data()
        
        print("\nSample data files generated successfully.")
        print("You can now run the dashboard with: streamlit run app.py")
        return True
    
    # Check for existing data files
    miifit_exists = os.path.exists('MiiFit.db') or os.path.exists('DataSnippet.csv')
    garmin_exists = os.path.exists('summarizedActivities.json') or os.path.exists('Activities.csv')
    
    if not miifit_exists and not garmin_exists:
        print("Warning: No data files found.")
        print("You can generate sample data with: python setup.py --generate-sample")
        return False
    
    if not miifit_exists:
        print("Warning: Neither MiiFit database (MiiFit.db) nor CSV file (DataSnippet.csv) found.")
    
    if not garmin_exists:
        print("Warning: Neither Garmin JSON file (summarizedActivities.json) nor CSV file (Activities.csv) found.")
    
    print("\nSetup check completed.")
    print("You can run the dashboard with: streamlit run app.py")
    return True

def update_config_for_sample_data():
    """Update the config.py file to use sample data files."""
    try:
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        # Update the paths
        config_content = config_content.replace(
            "MIIFIT_PATH = find_data_file(\"MiiFit.db\")", 
            "MIIFIT_PATH = find_data_file(\"DataSnippet.csv\")"
        )
        config_content = config_content.replace(
            "GARMIN_PATH = find_data_file(\"summarizedActivities.json\")", 
            "GARMIN_PATH = find_data_file(\"Activities.csv\")"
        )
        
        # Write back the updated config
        with open('config.py', 'w') as f:
            f.write(config_content)
        
        print("Updated config.py to use sample data files.")
    except Exception as e:
        print(f"Error updating config.py: {e}")


def detect_available_data_files():
    """Detect available data files and update config.py accordingly."""
    data_files = {
        'miifit_db': os.path.exists('MiiFit.db'),
        'miifit_csv': os.path.exists('DataSnippet.csv'),
        'garmin_json': os.path.exists('summarizedActivities.json'),
        'garmin_csv': os.path.exists('Activities.csv'),
        'polar_f11': os.path.exists('PolarF11Data.csv'),
        'charge_hr': os.path.exists('ChargeHRDataScrape.csv')
    }
    
    print("Available data files:")
    for file_type, exists in data_files.items():
        print(f"  {file_type}: {'Found' if exists else 'Not found'}")
    
    # Update config.py based on available files
    try:
        if not os.path.exists('config.py'):
            print("config.py not found, cannot update.")
            return
        
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        # Set paths based on available files
        if data_files['miifit_db']:
            miifit_path = 'MiiFit.db'
        elif data_files['miifit_csv']:
            miifit_path = 'DataSnippet.csv'
        else:
            miifit_path = 'MiiFit.db'  # Default
        
        if data_files['garmin_json']:
            garmin_path = 'summarizedActivities.json'
        elif data_files['garmin_csv']:
            garmin_path = 'Activities.csv'
        else:
            garmin_path = 'summarizedActivities.json'  # Default
        
        # Update config paths
        config_content = config_content.replace(
            "MIIFIT_PATH = find_data_file(\"MiiFit.db\")", 
            f"MIIFIT_PATH = find_data_file(\"{miifit_path}\")"
        )
        config_content = config_content.replace(
            "GARMIN_PATH = find_data_file(\"summarizedActivities.json\")", 
            f"GARMIN_PATH = find_data_file(\"{garmin_path}\")"
        )
        
        # Add/update paths for new data sources
        polar_path = 'PolarF11Data.csv' if data_files['polar_f11'] else 'PolarF11Data.csv'
        charge_path = 'ChargeHRDataScrape.csv' if data_files['charge_hr'] else 'ChargeHRDataScrape.csv'
        
        if "POLAR_F11_PATH" in config_content:
            config_content = config_content.replace(
                "POLAR_F11_PATH = find_data_file(\"PolarF11Data.csv\")", 
                f"POLAR_F11_PATH = find_data_file(\"{polar_path}\")"
            )
        
        if "CHARGE_HR_PATH" in config_content:
            config_content = config_content.replace(
                "CHARGE_HR_PATH = find_data_file(\"ChargeHRDataScrape.csv\")", 
                f"CHARGE_HR_PATH = find_data_file(\"{charge_path}\")"
            )
        
        # Write back the updated config
        with open('config.py', 'w') as f:
            f.write(config_content)
        
        print(f"Updated config.py to use:")
        print(f"  MiiFit: {miifit_path}")
        print(f"  Garmin: {garmin_path}")
        print(f"  PolarF11: {polar_path}")
        print(f"  ChargeHR: {charge_path}")
    except Exception as e:
        print(f"Error updating config.py: {e}")

if __name__ == "__main__":
    # Check available data files
    detect_available_data_files()
    
    # Run the normal configuration check
    check_configuration()
