"""Configuration settings for the fitness dashboard."""

import os
import logging

# Setup logging
logging.basicConfig(filename='dashboard.log', level=logging.DEBUG)

def find_data_file(filename):
    """Find data files in multiple locations with improved search logic.
    
    Search order:
    1. Exact path specified
    2. Current directory
    3. 'data' subdirectory 
    4. Parent directory
    5. Parent's 'data' subdirectory
    """
    # Check if the file exists as an absolute path
    if os.path.isabs(filename) and os.path.exists(filename):
        logging.debug(f"Found file at absolute path: {filename}")
        return filename
    
    # Extract just the filename if a path was provided
    base_filename = os.path.basename(filename)
    
    # List of locations to search
    search_locations = [
        filename,                                     # Original path as provided
        base_filename,                                # Just the filename in current dir
        os.path.join("data", base_filename),          # data/filename
        os.path.join("..", base_filename),            # ../filename
        os.path.join("..", "data", base_filename)     # ../data/filename
    ]
    
    # Search all locations
    for location in search_locations:
        if os.path.exists(location):
            logging.debug(f"Found file at: {location}")
            return location
    
    # If not found, at least log which locations were checked
    logging.debug(f"File {base_filename} not found. Searched in: {search_locations}")
    
    # Return original path (will be checked for existence later)
    return filename

# Data file paths - default values
DEFAULT_MIIFIT_PATH = "MiiFit.db"
DEFAULT_GARMIN_PATH = "summarizedActivities.json"
DEFAULT_POLAR_F11_PATH = "PolarF11Data.csv"
DEFAULT_CHARGE_HR_PATH = "ChargeHRDataScrape.csv"

# Apply search function to find actual file paths
MIIFIT_PATH = find_data_file("/home/blueaz/Downloads/SensorDownload/Current/MiiFit.db")
GARMIN_PATH = find_data_file("/home/blueaz/Downloads/SensorDownload/Current/summarizedActivities.json")
POLAR_F11_PATH = find_data_file("/home/blueaz/Downloads/SensorDownload/Current/PolarF11Data.csv")
CHARGE_HR_PATH = find_data_file("/home/blueaz/Downloads/SensorDownload/Current/ChargeHRDataScrape.csv")

# Fallback to checking for files in data directory with common names if not found
if not os.path.exists(MIIFIT_PATH):
    MIIFIT_PATH = find_data_file(DEFAULT_MIIFIT_PATH)
    
if not os.path.exists(GARMIN_PATH):
    # Try alternative filename conventions
    alternatives = [DEFAULT_GARMIN_PATH, "garmin_activities.json", "Activities.csv", "garmin_data.csv"]
    for alt in alternatives:
        path = find_data_file(alt)
        if os.path.exists(path):
            GARMIN_PATH = path
            break
            
if not os.path.exists(POLAR_F11_PATH):
    POLAR_F11_PATH = find_data_file(DEFAULT_POLAR_F11_PATH)
    
if not os.path.exists(CHARGE_HR_PATH):
    CHARGE_HR_PATH = find_data_file(DEFAULT_CHARGE_HR_PATH)

# Debug mode flag
DEBUG = True

# Column mappings for data standardization
MIIFIT_COLUMNS = {
    'DATE': 'date',
    'TYPE': 'activity_type',
    'AVGHR': 'avg_hr',
    'MAX_HR': 'max_hr',
    'CAL': 'calories',
    'duration_minutes': 'duration'
}

GARMIN_COLUMNS = {
    'Date': 'date',
    'Activity Type': 'activity_type',
    'Avg HR': 'avg_hr',
    'Max HR': 'max_hr',
    'Calories': 'calories',
    'Total Time': 'duration_string'  # Will be converted to minutes
}

POLAR_F11_COLUMNS = {
    'sport': 'activity_type',
    'time': 'date',
    'calories': 'calories',
    'Duration': 'duration_string',
    'average': 'avg_hr',
    'maximum': 'max_hr',
    'Location': 'location',
    'Cal/HR': 'cal_hr_ratio'  # New column that might be useful
}

CHARGE_HR_COLUMNS = {
    'Date': 'date',  # Changed from FullDate to Date
    'Activity': 'activity_type',
    'Cals': 'calories',
    'Duration': 'duration_string',
    'Location': 'location',
    'Year': 'year'
}

# Activity type mapping for standardization
ACTIVITY_TYPE_MAPPING = {
    # MiiFit mappings
    'Free': 'Free Workout',
    'IndCyc': 'Indoor Cycling',
    'OutCyc': 'Outdoor Cycling',
    'Elliptical': 'Elliptical',
    'Yoga': 'Yoga',
    'Swim': 'Swimming',
    
    # Garmin mappings
    'Cycling': 'Outdoor Cycling',
    'CYCLING': 'Outdoor Cycling',
    'CYCLING_INDOOR': 'Indoor Cycling',
    'Indoor Cycling': 'Indoor Cycling',
    'Running': 'Running',
    'RUNNING': 'Running',
    'Swimming': 'Swimming',
    'SWIMMING': 'Swimming',
    'POOL_SWIMMING': 'Swimming',
    'Walking': 'Walking',
    'WALKING': 'Walking',
    'Yoga': 'Yoga',
    'YOGA': 'Yoga',
    'Strength Training': 'Strength Training',
    'STRENGTH_TRAINING': 'Strength Training',
    'Elliptical': 'Elliptical',
    'ELLIPTICAL': 'Elliptical',
    'CARDIO': 'Cardio',
    'Cardio': 'Cardio',
    'TREADMILL_RUNNING': 'Running',
    'OPEN_WATER_SWIMMING': 'Swimming',
    'FITNESS_EQUIPMENT': 'Gym Workout',
    
    # Polar F11 and Charge HR mappings
    'Tennis': 'Tennis',
    'Biking': 'Outdoor Cycling',
    'treadmill': 'Running',
    'Gym': 'Gym Workout',
    'Spin': 'Indoor Cycling',  # From PolarF11 example
    'Mountain Biking': 'Outdoor Cycling'  # From ChargeHR example
}

# Plot settings
PLOT_HEIGHT = 600
PLOT_WIDTH = 800

# Color schemes
COLOR_SCHEME_SOURCES = {
    'MiiFit': '#1f77b4',
    'Garmin': '#ff7f0e',
    'PolarF11': '#2ca02c',
    'ChargeHR': '#d62728'
}

COLOR_SCHEME = COLOR_SCHEME_SOURCES  # For backward compatibility with original dashboard

COLOR_SCHEME_ACTIVITIES = {
    'Free Workout': '#1f77b4',
    'Indoor Cycling': '#ff7f0e',
    'Outdoor Cycling': '#2ca02c',
    'Elliptical': '#d62728',
    'Yoga': '#9467bd',
    'Swimming': '#8c564b',
    'Running': '#e377c2',
    'Walking': '#7f7f7f',
    'Strength Training': '#bcbd22',
    'Cardio': '#17becf',
    'Gym Workout': '#9467bd',
    'Tennis': '#e7ba52'
}

# Date format settings
DATE_FORMAT = '%Y-%m-%d'

# Default time ranges for date selector
DEFAULT_DAYS_BACK = 30  # Default to last 30 days to match original dashboard

# Check at startup which files were found
logging.info("Data files found:")
logging.info(f"MiiFit: {MIIFIT_PATH} (exists: {os.path.exists(MIIFIT_PATH)})")
logging.info(f"Garmin: {GARMIN_PATH} (exists: {os.path.exists(GARMIN_PATH)})")
logging.info(f"PolarF11: {POLAR_F11_PATH} (exists: {os.path.exists(POLAR_F11_PATH)})")
logging.info(f"ChargeHR: {CHARGE_HR_PATH} (exists: {os.path.exists(CHARGE_HR_PATH)})")

# Print to console during development if in debug mode
if DEBUG:
    print("Data files found:")
    print(f"MiiFit: {MIIFIT_PATH} (exists: {os.path.exists(MIIFIT_PATH)})")
    print(f"Garmin: {GARMIN_PATH} (exists: {os.path.exists(GARMIN_PATH)})")
    print(f"PolarF11: {POLAR_F11_PATH} (exists: {os.path.exists(POLAR_F11_PATH)})")
    print(f"ChargeHR: {CHARGE_HR_PATH} (exists: {os.path.exists(CHARGE_HR_PATH)})")
