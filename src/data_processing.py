import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder,StandardScaler # Encoding categorical variables Using OneHotEncoder and StandardScaler for scaling in line 46
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from xverse.transformer import WOE


# read the data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) 
    return df


# Aggregate per customer
def aggregate_customer_feature(df: pd.DataFrame)->pd.DataFrame:
    customer_agg = (
        df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            txn_count=("TransactionId", "count"),
            std_amount=("Amount", "std")
        )
        .reset_index()
    )
    return customer_agg


# Date-time features
def create_datetime_features(df: pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    df['txn_hour'] = df['TransactionStartTime'].dt.hour
    df['txn_day'] = df['TransactionStartTime'].dt.day
    df['txn_month'] = df['TransactionStartTime'].dt.month
    df['txn_year'] = df['TransactionStartTime'].dt.year

    return df


# RFM CALCULATION(Recency, Frequency, Monetary (RFM) metrics)
def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    return rfm


# RFM CLUSTERING + TARGET VARIABLE(Cluster customers using RFM and define high-risk proxy target)
def create_risk_target(rfm: pd.DataFrame) -> pd.DataFrame:
    # Scale RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (worst engagement)
    cluster_summary = (
        rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )

    # High Recency + Low Frequency + Low Monetary
    risk_cluster = cluster_summary.sort_values(
        by=["Recency", "Frequency", "Monetary"],
        ascending=[False, True, True],
    ).iloc[0]["cluster"]

    # Create binary target
    rfm["is_high_risk"] = (rfm["cluster"] == risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]



# Build sklearn preprocessing pipeline
def build_feature_pipeline(numeric_features, categorical_features):
    
    numeric_pipeline = Pipeline(
        # imputation for handling missing values and scaler for scaling (z = x-mean/std) for all 
        steps=[
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        # imputation for handling missing values and encoder for converting categorical values into binary columns 
        steps =[
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        # combine pipelines
        transformers=[
            ("num",numeric_pipeline,numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor



# =========================================================
# End-to-end feature engineering function
def process_data(raw_path : str, output_path: str) -> pd.DataFrame:
    # 1. load data
    df = load_data(raw_path)

    # 2. date time feature
    df = create_datetime_features(df)



    # TASK-3: FEATURE ENGINEERING
    # 3.  Aggregate customer features
    customer_features = aggregate_customer_feature(df)

    # 4. Select categorical features (customer-level)
    categorical_cols = [
        "ProductCategory",
        "ChannelId",
        "ProviderId",
        "PricingStrategy",
    ]

    categorical_df = (
        df.groupby("CustomerId")[categorical_cols]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    # 5.Merge aggregated numerical + categorical
    final_df = customer_features.merge(
        categorical_df, on="CustomerId", how="left"
    )

    # 6. Define feature lists
    numeric_features = [
        "total_amount",
        "avg_amount",
        "txn_count",
        "std_amount",
    ]

    categorical_features = categorical_cols

    #7. Build preprocessing pipeline
    preprocessor = build_feature_pipeline(
        numeric_features, categorical_features
    )

    # 8. Fit & transform data
    processed_array = preprocessor.fit_transform(final_df)

    # 9. Create column names
    encoded_cat_cols = preprocessor.named_transformers_[
        "cat"
    ].named_steps["encoder"].get_feature_names_out(categorical_features)

    all_columns = numeric_features + list(encoded_cat_cols)

    processed_df = pd.DataFrame(
        processed_array, columns=all_columns
    )

    # 10. Add CustomerId back
    processed_df["CustomerId"] = final_df["CustomerId"].values


    # TASK-4: RFM + TARGET
    # 11
    rfm = calculate_rfm(df)
    target = create_risk_target(rfm)

    # 12. Merge features + target
    processed_df = processed_df.merge(target, on="CustomerId", how="left")

    # 13. Save processed dataset
    processed_df.to_csv(output_path, index=False)

    return processed_df


# Apply Weight of Evidence transformation
def apply_woe(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    woe = WOE()
    X_woe = woe.fit_transform(X,y)
    return X_woe