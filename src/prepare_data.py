import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_raw_data
from src.data.splitter import create_time_splits, save_splits
from src.utils.config_data import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR,
    REFERENCE_RATIO,
    VALIDATION_RATIO,
    NUM_BATCHES,
    TIME_COLUMN,
    LABEL_COLUMN
)

def main():
    print("=" * 60)
    print("Credit Card Fraud - Data Preparation Pipeline")
    print("=" * 60)
    
    # 1. Load raw data
    raw_file = RAW_DATA_DIR / "creditcard.csv"
    print(f"\nLoading raw data from: {raw_file}")
    
    # Pass TIME_COLUMN to the loader as refactored previously
    df = load_raw_data(raw_file, time_column=TIME_COLUMN)
    
    print(f"Loaded {len(df)} samples")
    # Use LABEL_COLUMN and TIME_COLUMN from config instead of hardcoded strings
    print(f"  - Fraud ratio: {df[LABEL_COLUMN].mean():.4f}")
    print(f"  - Time range: {df[TIME_COLUMN].min():.0f} - {df[TIME_COLUMN].max():.0f}")
    
    # 2. Create splits
    print(f"\nCreating time-based splits...")
    print(f"  - Reference: {REFERENCE_RATIO*100:.0f}%")
    print(f"  - Validation: {VALIDATION_RATIO*100:.0f}%")
    print(f"  - Production batches: {NUM_BATCHES}")
    
    reference, validation, batches = create_time_splits(
        df,
        reference_ratio=REFERENCE_RATIO,
        validation_ratio=VALIDATION_RATIO,
        num_batches=NUM_BATCHES
    )
    
    # 3. Save splits
    print(f"\nSaving processed data to: {PROCESSED_DATA_DIR}")
    save_splits(reference, validation, batches, PROCESSED_DATA_DIR)
    

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()