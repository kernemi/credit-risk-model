import os
from src.data_processing import process_data

def test_process_data_creates_file():
    raw_path = "data/raw/data.csv"         
    output_file ="data/processed/features.csv"  

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Run the pipeline
    processed_df = process_data(raw_path, str(output_file))
   
    # Check that file exists
    assert os.path.exists(output_file), "Processed file not found!"

    # Check that target column exists
    assert "is_high_risk" in processed_df.columns, "Target column missing!"
    
    # Check some basic properties
    assert processed_df.shape[0] > 0, "No rows in processed data"
    assert processed_df.shape[1] > 1, "No columns in processed data"

    print(f"Processed data successfully saved to {output_file}")