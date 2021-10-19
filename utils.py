import uuid
import streamlit as st
import os
from google.cloud import storage


# Bucket ID for Google Storage upload
if os.environ.get("TEST_NUTRIFY_ENV_VAR"):
    BUCKET_ID = "food-vision-images-test-upload"  # test bucket
    print(f"***Using test Google Storage bucket: {BUCKET_ID}***")
else:
    BUCKET_ID = "food-vision-images"  # prod bucket


def upload_blob(source_file_name, destination_blob_name):
    """
    Uploads image file to Google Storage bucket.
    """
    # Path to file
    # source_file_name =
    # ID of GCS object to store
    print("Starting to try and upload...")

    # Authenticate storage,
    # see: https://cloud.google.com/docs/authentication/production#passing_code
    storage_client = storage.Client.from_service_account_info(
        st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    bucket = storage_client.bucket(bucket_name=BUCKET_ID)
    blob = bucket.blob(destination_blob_name)

    # Upload object
    blob.upload_from_file(source_file_name, rewind=True)

    print(f"File {source_file_name} uploaded to {destination_blob_name}")


def create_unique_filename() -> str:
    """
    Creates a unique filename for storing uploaded images.
    """
    return str(uuid.uuid4())
