import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Google Cloud Storage setup
BUCKET_NAME = "doge_bucket45"
PREDICTIONS_LOG_CSV = "predictions_log.csv"
LABELS_FEATURES_CSV = "labels_metrics_sampled.csv"

# Function to download a file from GCP bucket
def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

# Download the required files
download_from_gcs(BUCKET_NAME, PREDICTIONS_LOG_CSV, PREDICTIONS_LOG_CSV)
download_from_gcs(BUCKET_NAME, LABELS_FEATURES_CSV, LABELS_FEATURES_CSV)

# Load the CSV files
predictions_log = pd.read_csv(PREDICTIONS_LOG_CSV)
labels_features = pd.read_csv(LABELS_FEATURES_CSV)

# Filter numeric columns only
numeric_columns_predictions = predictions_log.select_dtypes(include=["float64", "int64"]).columns
numeric_columns_labels = labels_features.select_dtypes(include=["float64", "int64"]).columns

# Ensure both datasets have the same numeric columns
common_columns = list(set(numeric_columns_predictions).intersection(numeric_columns_labels))



# Function to upload a file to GCS bucket
def upload_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """
    Uploads a file to a GCS bucket.

    :param bucket_name: The name of the GCS bucket
    :param source_file_name: The local path of the file to be uploaded
    :param destination_blob_name: The destination path within the bucket
    """
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

# Upload the generated report.html to the bucket
REPORT_FILENAME = "report.html"
DESTINATION_BLOB_NAME = "data_drift_reports/report.html"  # Path within the bucket


# Drop zero-variance columns
def drop_constant_columns(df):
    return df.loc[:, (df.std() != 0)]

predictions_log_filtered = drop_constant_columns(predictions_log[common_columns])
labels_features_filtered = drop_constant_columns(labels_features[common_columns])

# Ensure consistency between datasets
common_columns_filtered = list(set(predictions_log_filtered.columns).intersection(labels_features_filtered.columns))
predictions_log_filtered = predictions_log_filtered[common_columns_filtered]
labels_features_filtered = labels_features_filtered[common_columns_filtered]

# Standardize the numeric features
scaler = StandardScaler()
predictions_log_scaled = pd.DataFrame(
    scaler.fit_transform(predictions_log_filtered),
    columns=common_columns_filtered
)
labels_features_scaled = pd.DataFrame(
    scaler.transform(labels_features_filtered),
    columns=common_columns_filtered
)

# Create a data drift report
report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=labels_features_scaled, current_data=predictions_log_scaled)
report.save_html("report.html")
upload_to_gcs(BUCKET_NAME, REPORT_FILENAME, DESTINATION_BLOB_NAME)

print("Data drift report generated and saved as report.html.")
