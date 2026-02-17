import pandas as pd
from pathlib import Path

def load_raw_data(filepath: Path, time_column: str = "Time"):
    """
    Load raw credit card fraud dataset.
    
    Args:
        filepath: Path to raw CSV file
        
    Returns:
        DataFrame with sorted data by Time
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data not found at {filepath}")

    df = pd.read_csv(filepath)
    
    # Sort by time 
    df = df.sort_values(time_column).reset_index(drop=True)
    
    return df

