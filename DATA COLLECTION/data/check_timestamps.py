import pandas as pd
from datetime import datetime, timedelta

def check_missing_timestamps(csv_file, timestamp_column='timestamp', time_interval=5):
    """
    Check for missing timestamps in a CSV file where timestamps should be 5 minutes apart.
    
    Args:
        csv_file (str): Path to the CSV file
        timestamp_column (str): Name of the timestamp column
        time_interval (int): Expected time interval in minutes between timestamps
    
    Returns:
        list: List of missing timestamps
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp column to datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Sort the dataframe by timestamp
    df = df.sort_values(by=timestamp_column)
    
    # Initialize list to store missing timestamps
    missing_timestamps = []
    
    # Check for missing timestamps
    for i in range(len(df) - 1):
        current_time = df[timestamp_column].iloc[i]
        next_time = df[timestamp_column].iloc[i + 1]
        
        # Calculate expected next timestamp
        expected_next_time = current_time + timedelta(minutes=time_interval)
        
        # If there's a gap larger than the expected interval
        if next_time > expected_next_time:
            # Add all missing timestamps in the gap
            while expected_next_time < next_time:
                missing_timestamps.append(expected_next_time)
                expected_next_time += timedelta(minutes=time_interval)
    
    return missing_timestamps

if __name__ == "__main__":
    # Example usage
    csv_file = "derivs_hist/open-interest.csv"  # Replace with your CSV file path
    missing_timestamps = check_missing_timestamps(csv_file)
    
    if missing_timestamps:
        print(f"Found {len(missing_timestamps)} missing timestamps:")
        for timestamp in missing_timestamps:
            print(timestamp)
    else:
        print("No missing timestamps found.") 