import pandas as pd
from datetime import timedelta
import os

def add_hours_to_datetime(input_file, output_file=None, hours=4):
    """
    Add specified hours to datetime values in a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the modified CSV file. 
                                    If None, will append '_modified' to input filename
        hours (int, optional): Number of hours to add. Default is 4
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert all datetime-like columns
    for col in df.columns:
        try:
            # Try to convert column to datetime
            df[col] = pd.to_datetime(df[col])
            # Add hours to datetime values
            df[col] = df[col] + timedelta(hours=hours)
        except:
            # Skip if column can't be converted to datetime
            continue
    
    # Generate output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_modified{ext}"
    
    # Save the modified dataframe
    df.to_csv(output_file, index=False)
    print(f"Modified data saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    input_file = "missing_timestamps.csv"  # Replace with your input file path
    add_hours_to_datetime(input_file) 