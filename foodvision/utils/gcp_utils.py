"""
Utility functions for working with Google Cloud Platform (GCP) services.
"""
import os


from google.cloud import storage

# TODO: Create Google Service Account credentials - is this better than env variables?
# from google.oauth2 import service_account
# credentials = service_account.Credentials.from_service_account_file("your-json-path-with-filename.json")

def set_gcp_credentials(path_to_key=None):
    """Sets GCP credentials from environment variable.
    Raises:
        RuntimeError: If environment variable is not set, it will raise a RuntimeError.
    """
    if path_to_key is not None:
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_key
            print("[INFO] GCP credentials set!")
        except:
            raise RuntimeError(
                "GCP credentials not set, check GOOGLE_APPLICATION_CREDENTIALS"
            )
    else:
        raise ValueError("Path to GCP key not provided, check path_to_key argument and provide a path to a Google Storage key as JSON.")

def test_gcp_connection():
    """Tests connection to GCP based on the presense of an environment variable.
    Raises:
        RuntimeError: If connection can't be made, it will raise a RuntimeError.
    """
    try:
        storage.Client()
        print(
            "[INFO] GCP connection successful! Access to GCP for saving/loading data and models available."
        )
    except:
        raise RuntimeError(
            "GCP connection unsuccessful, this is required for storing data and models, check GOOGLE_APPLICATION_CREDENTIALS"
        )

def get_list_of_blobs(bucket_name, names_only=False, prefix=None, delimiter=None):
    """Gets a list of blobs from a target Google Storage bucket

    Args:
        bucket_name (str): Target Google Storage bucket name.
        names_only (bool, optional): If True, returns a list of blob names only not the actual blob. Defaults to False.
        prefix (str, optional): Filepath on Google Storage bucket to search (e.g. bucket_name/prefix/to/target/dir). Defaults to None.
        delimiter (str, optional): Use "/" to get files in a target directory. Defaults to None.

    Returns:
        list: a list of blob names or actual blobs
    """

    storage_client = storage.Client()

    # TODO: turn this into a function like here: https://cloud.google.com/storage/docs/listing-objects#list-objects 
    blobs = storage_client.list_blobs(bucket_or_name=bucket_name,
                                      prefix=prefix,
                                      delimiter=delimiter)

    # Return file names or actual blobs?
    if names_only:
        return [blob.name for blob in blobs]
    else:
        return list(blobs)


def download_blobs_to_file(blobs, destination_dir):
    """Downloads a list of blobs to a destination directory

    Args:
        blobs (list): List of Google Cloud Storage blobs.
        destination_dir (str): Destination directory to download blobs to.

    Returns:
        _type_: _description_
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Loop through blobs and download them
    num_downloaded = 0
    for blob in blobs:

        # Get the filename of the blob
        blob_filename = blob.name.split("/")[-1]

        # Setup a destination for the blob
        blob_file_destination = os.path.join(destination_dir, blob_filename)
        
        try:
            blob.download_to_filename(blob_file_destination)
            print(f"[INFO] Downloading {blob.name} to {blob_file_destination}")
            num_downloaded += 1
        except Exception as e:
            print(f"Error downloading {blob.name} to {blob_file_destination}, error: {e}")
    
    # Print the number of blobs downloaded
    print(f"[INFO] Number of blobs downloaded: {num_downloaded}")
    print(f"[INFO] Total files in {destination_dir}: {len(os.listdir(destination_dir))}")

def upload_to_gs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    print(f"[INFO] Uploading {source_file_name} to {destination_blob_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"[INFO] Connected to Google Storage bucket: {bucket_name}")
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"[INFO] File {source_file_name} uploaded to {bucket_name}/{destination_blob_name}.")
    print(f"[INFO] File size: {blob.size} bytes")

    # TODO: Make the blob public -- do I want this to happen?
    # blob.make_public()
    # print(f"[INFO] Blob public URL: {blob.public_url}")
    # print(f"[INFO] Blob download URL: {blob._get_download_url()}")

    return destination_blob_name

def rename_blob(bucket_name, blob_name, new_name):
    """Renames a blob."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of the GCS object to rename
    # blob_name = "your-object-name"
    # The new ID of the GCS object
    # new_name = "new-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    new_blob = bucket.rename_blob(blob, new_name)

    print(f"[INFO] Blob {blob.name} has been renamed to {new_blob.name}")

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print(f"Blob {blob_name} deleted.")