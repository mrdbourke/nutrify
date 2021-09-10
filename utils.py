import uuid
import streamlit as st
import os
from google.cloud import storage


### Uploader function to Google Storage ###
# Bucket ID for Google Storage upload
BUCKET_ID = "food-vision-images"

def upload_blob(source_file_name, destination_blob_name):
    """
    Uploads image file to Google Storage bucket. 
    """
    # Path to file
    # source_file_name = 
    # ID of GCS object to store
    print("Starting to try and upload...")

    # Authenticate storage, see: https://cloud.google.com/docs/authentication/production#passing_code 
    storage_client = storage.Client.from_service_account_info(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    bucket = storage_client.bucket(bucket_name=BUCKET_ID)
    blob = bucket.blob(destination_blob_name)

    # Upload object
    blob.upload_from_file(source_file_name, rewind=True)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}"
    )

### Create a unique file name ID for saving images ###
def create_unique_filename():
    """
    Creates a unique filename for storing uploaded images.
    """
    return str(uuid.uuid4())