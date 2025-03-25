# Fitness Dashboard

A comprehensive Streamlit dashboard for visualizing and analyzing fitness tracking data from multiple sources (MiiFit, Garmin, PolarF11, and ChargeHR).

## Overview

This application provides interactive visualizations of fitness metrics like heart rate, activity duration, calories burned, and more. It automatically locates and processes data files from multiple fitness tracking devices, allowing users to analyze their fitness patterns over time.

## Features

- **Automatic Data Detection**: Searches for data files in multiple locations without requiring manual file selection
- **Multiple Data Source Support**:
  - MiiFit (.db/.csv)
  - Garmin (.json/.csv)
  - PolarF11 (.csv)
  - ChargeHR (.csv)
- **Interactive Filtering**:
  - Date range selection
  - Data source selection
  - Activity type filtering
- **Comprehensive Visualizations**:
  - Quick Overview Dashboard
  - Heart Rate Analysis
  - Activity Distribution
  - Duration Analysis
  - Metric Relationships
  - Raw Data Access
- **Sample Data Generation**: Option to use generated sample data for testing

## Installation

### Prerequisites
- Python 3.11+ recommended
- Required packages: streamlit, pandas, plotly, numpy

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fitness-dashboard.git
cd fitness-dashboard
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place your fitness data files in one of these locations:
   - Directly in the project directory
   - In a subdirectory called `data/`
   - Or use the original paths specified in the code

## Data File Structure

The dashboard looks for the following files:

### MiiFit
- `MiiFit.db` (SQLite database) or `DataSnippet.csv`
- Required columns: DATE, TYPE, AVGHR, MAX_HR, CAL, duration_minutes

### Garmin
- `summarizedActivities.json` or `Activities.csv` 
- Required columns: Date, Activity Type, Avg HR, Max HR, Calories, Total Time

### PolarF11
- `PolarF11Data.csv`
- Required columns: sport, time, calories, Duration, average, maximum

### ChargeHR
- `ChargeHRDataScrape.csv`
- Required columns: Date, Activity, Cals, Duration

## Running the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will automatically search for data files in the configured locations. If no files are found, it will offer to use sample data for testing.

## Dashboard Tabs

### Quick Overview
A simplified view showing summary statistics, heart rate trends, activity distribution, and key metric relationships.

### Heart Rate Analysis
Detailed analysis of heart rate trends over time and by activity type.

### Activity Distribution
Breakdowns of activity types, trends over time, and calorie distribution.

### Duration Analysis
Activity duration patterns, including daily and weekly summaries.

### Metric Relationships
Correlation analysis between different fitness metrics.

### Raw Data
Access to the underlying dataset with download capability.

## File Structure

- `streamlit_app.py`: Main application with all dashboard functionality
- `requirements.txt`: Required Python packages
- `data/`: Recommended directory for storing fitness data files
- `README.md`: This documentation file

## Customization

You can customize the dashboard by modifying:
- Default file paths in the load_*_data() functions
- Color schemes in the COLOR_SCHEME_* variables
- The file search locations in the find_data_file() function

## Troubleshooting

If you encounter issues:

1. **No data found**: Check that your files are in one of the searched locations and have the correct format/extension
2. **File format errors**: Verify that your CSV files have the expected column names
3. **Python version**: The dashboard works best with Python 3.11+
4. **Sample data**: Use the "Use sample data" checkbox in the sidebar to test functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.
