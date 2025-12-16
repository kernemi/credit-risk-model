import pandas as pd
from src.data_processing import process_data

def test_feature_columns():
    df = process_data(
        raw_path="data/raw/data.csv",
        output_path="data/processed/features_test.csv",
    )
    expected_columns = [
        "total_amount", "avg_amount", "txn_count", "std_amount",
        "CustomerId", "is_high_risk"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

def test_target_variable():
    df = process_data(
        raw_path="data/raw/data.csv",
        output_path="data/processed/features_test.csv",
    )
    assert df["is_high_risk"].nunique() == 2, "Target variable must be binary"
