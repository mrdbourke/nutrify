# TODO: add code from 04-merge-labels-from-label-studio.ipynb
import json
import os

from pathlib import Path

import pandas as pd
import wandb

from tqdm.auto import tqdm

# Import config
from configs.default_config import config

# Connect to GCP
from utils.gcp_utils import set_gcp_credentials, test_gcp_connection
set_gcp_credentials(path_to_key="utils/google-storage-key.json")
test_gcp_connection()

# Setup variables
WANDB_PROJECT = config.wandb_project
WANDB_RUN_TAGS = ["update_and_merge_manual_labels"]
WANDB_JOB_TYPE = "merge_labels_from_label_studio"
WANDB_RUN_NOTES = "Update and merge manual labels from Label Studio"

WANDB_MODEL = config.wandb_model_artifact
WANDB_DATASET = config.wandb_dataset_artifact
WANDB_LABELS = config.wandb_labels_artifact

PRETRAINED = config.pretrained

MODEL = config.model
INPUT_SIZE = config.input_size

# Import most wrong labels from Label Studio (these are in a Google Cloud Storage bucket)
GS_BUCKET_NAME = config.gs_bucket_name
GS_CLASSIFICATION_LABELS_TO_FIX_PREFIX = "label_studio_exports/classification/"

# Setup local directories to download to
LABELS_TO_FIX_DIR = "labels_to_fix"
LABELS_TO_FIX_DOWNLOAD_DIR = os.path.join(LABELS_TO_FIX_DIR, GS_CLASSIFICATION_LABELS_TO_FIX_PREFIX)

# Get list of blobs and download them
from utils.gcp_utils import get_list_of_blobs, download_blobs_to_file

list_of_label_update_blobs = get_list_of_blobs(bucket_name=GS_BUCKET_NAME,
                          prefix=GS_CLASSIFICATION_LABELS_TO_FIX_PREFIX,
                          delimiter="/")

if len(list_of_label_update_blobs) > 0:
    print(f"[INFO] Found {len(list_of_label_update_blobs)} labels to update in bucket: '{os.path.join(GS_BUCKET_NAME, GS_CLASSIFICATION_LABELS_TO_FIX_PREFIX)}' (downloading to '{LABELS_TO_FIX_DOWNLOAD_DIR}')...")
    download_blobs_to_file(blobs=list_of_label_update_blobs,
                           destination_dir=LABELS_TO_FIX_DOWNLOAD_DIR)
else:
    print(f"[INFO] No labels found in bucket: '{os.path.join(GS_BUCKET_NAME, GS_CLASSIFICATION_LABELS_TO_FIX_PREFIX)}', perhaps try labelling some data with `fix_labels.py` and then rerun this script? Exiting...")
    exit()

# Connect to Weights & Biases
from utils.wandb_utils import wandb_load_artifact, wandb_download_and_load_labels

run = wandb.init(project=WANDB_PROJECT, 
                 job_type=WANDB_JOB_TYPE,
                 tags=WANDB_RUN_TAGS,
                 notes=WANDB_RUN_NOTES)

images_dir = wandb_load_artifact(
    wandb_run=run, 
    artifact_name=WANDB_DATASET, 
    artifact_type="dataset")

annotations, class_names, class_dict, reverse_class_dict, labels_path = wandb_download_and_load_labels(wandb_run=run,
wandb_labels_artifact_name=WANDB_LABELS)

# Make a copy of the original annotations for comparison later
original_annotations = annotations.copy()

# Format label studio JSON
def format_label_studio_json(label_studio_json):
    """Formats a Label Studio style JSON object into a dictionary compatible with the current label schema"""

    updated_image_and_label_dict = {}

    # Get the output result from Label Studio's labelling format
    updated_label_results = label_studio_json["result"]

    # Get the target image
    target_image = label_studio_json["task"]["data"]["image"]
    image_name = Path(target_image).name

    # Get the updated label (this is the class_name)
    updated_label = updated_label_results[0]["value"]["choices"][0]

    # Get different labelling parameters 
    try:
        clear_or_confusing = updated_label_results[1]["value"]["choices"][0]
        whole_food_or_dish = updated_label_results[2]["value"]["choices"][0]
        one_food_or_multiple = updated_label_results[3]["value"]["choices"][0]
    except IndexError:
        clear_or_confusing = None
        whole_food_or_dish = None
        one_food_or_multiple = None
    
    # Populate updated label dictionary
    updated_image_and_label_dict["image_path_on_gcp"] = target_image
    updated_image_and_label_dict["image_name"] = image_name
    updated_image_and_label_dict["updated_label"] = updated_label

    # TODO: create these from the values of the sections themselves rather than typing the actual name?
    updated_image_and_label_dict["clear_or_confusing"] = clear_or_confusing
    updated_image_and_label_dict["whole_food_or_dish"] = whole_food_or_dish
    updated_image_and_label_dict["one_food_or_multiple"] = one_food_or_multiple

    # Add a label last updated at timestamp
    import datetime
    updated_image_and_label_dict["label_last_updated_at"] = datetime.datetime.now()

    # Add a label source
    updated_image_and_label_dict["label_source"] = "manual_label_studio" # TODO: make this an arg?

    return updated_image_and_label_dict



def turn_labels_to_fix_to_list_of_dicts(path_to_labels_to_fix_dir):
    """Turns a directory of label_studio_export.json files into a list of dictionaries"""
    label_studio_export_paths = list(Path(path_to_labels_to_fix_dir).glob("*"))

    assert len(label_studio_export_paths) > 0, f"No label_studio_export_paths found, perhaps there's no updated labels? Check the bucket: {os.path.join(GS_BUCKET_NAME, GS_CLASSIFICATION_LABELS_TO_FIX_PREFIX)}, if no labels are presents, then there's nothing to update. Try running fix_labels.py and annotating some more images to fix."

    # Import the first label_studio_export_path as JSON
    import json
    updated_labels_to_fix_json = []
    for label_studio_export_path in label_studio_export_paths:
        try:
            with open(label_studio_export_path, "r") as f:
                label_studio_export = json.load(f)
                # Turn the label_studio_export JSON into a dictionary compatible with the current labels
                label_studio_export_formatted = format_label_studio_json(label_studio_export)
                updated_labels_to_fix_json.append(label_studio_export_formatted)
        except Exception as e:
            print(f"Error loading {label_studio_export_path} as JSON, error: {e}")
    
    # Format the label_studio_export JSON into a dictionary compatible with the current labels
        
    return updated_labels_to_fix_json

## TODO: fix below here to make it clearer
## TODO: Should almost be agnostic to the label schema (as in, pass any two DataFrames or something and it'll do the intersection and update)
updated_labels_to_fix = turn_labels_to_fix_to_list_of_dicts(path_to_labels_to_fix_dir=LABELS_TO_FIX_DOWNLOAD_DIR)
updated_labels_df = pd.DataFrame(updated_labels_to_fix)

# Set annotations index to image_name and updated_labels_df index to image_name
annotations.set_index("image_name", inplace=True)
updated_labels_df.set_index("image_name", inplace=True)

# Find the columns which occur in both dfs
intersecting_columns = list(annotations.columns.intersection(updated_labels_df.columns))
print(intersecting_columns)

# Set the updated_annotations df to the annotations df with the updated labels
updated_annotations = annotations.copy()

# Find the indexes where annotations and updated_labels_df have the same image_name (these are the values to update, e.g. img_1234 (old details) -> img_1234 (new details))
updated_annotations_indexes = updated_annotations.index.intersection(updated_labels_df.index) 

# Set the values in updated_annotations to the values in updated_labels_df (where the image_name matches)
updated_annotations.loc[updated_annotations_indexes, intersecting_columns] = updated_labels_df.loc[updated_annotations_indexes, intersecting_columns]

# Find the columns which are in updated_labels_df but not in annotations
updated_labels_df_columns_not_in_annotations = list(updated_labels_df.columns.difference(annotations.columns))
print(f"[INFO] Updated labels columns not in original annotations: {updated_labels_df_columns_not_in_annotations}")

# Add the updated_labels_df columns which are not in annotations (these are universal to all images)
updated_annotations[updated_labels_df_columns_not_in_annotations] = updated_labels_df[updated_labels_df_columns_not_in_annotations]

# Add a column for the updated_label_key mapping the class_dict to the updated_label
updated_annotations["updated_label_key"] = updated_annotations["updated_label"].map(reverse_class_dict)

# Get the rows where updated_label is not null and class_name is different to updated_label
updated_rows = updated_annotations[(updated_annotations["updated_label"].notnull()) & (updated_annotations["class_name"] != updated_annotations["updated_label"])]
num_updated_rows_with_new_class_label = len(updated_rows)

# Replace the class_name column with the updated labels as long as isna() is False
updated_annotations["class_name"] = updated_annotations["updated_label"].where(updated_annotations["updated_label"].isna() == False, updated_annotations["class_name"])

# Make sure all the label keys match the class_name
updated_annotations["label"] = updated_annotations["class_name"].map(reverse_class_dict)

# Get the number of rows that have different class_names values between updated_annotations and annotations
assert len(updated_rows) == len(updated_annotations[updated_annotations["class_name"] != annotations["class_name"]]), "Number of rows in `updated_rows` dataframe does not match the number of rows where `class_name` is different between `updated_annotations` and `annotations`, potential labelling mixup"
assert len(annotations) == len(updated_annotations), "Length of updated_annotations not the same as annotations, did you remove or add any samples?"

# Make sure all of the label values where updated_label is not null are the same as updated_label_key
assert list(updated_annotations[(updated_annotations["updated_label"].notnull())]["label"]) == list(updated_annotations[(updated_annotations["updated_label"].notnull())]["updated_label_key"]), "'label' values differ from 'updated_label_key' values where 'updated_label' is not null, potential labelling mixup"

# Reset the index
updated_annotations_reset_index = updated_annotations.reset_index()

# TODO: move this into utils folder 
def check_for_differences_between_df(df1, df2, columns_to_exclude: list=None):
    """Checks for differences between two dataframes, returns the number of differences"""
    # Find the intersection of the columns
    intersecting_columns = list(df1.columns.intersection(df2.columns))

    # Remove columns_to_exclude from intersecting_columns
    if columns_to_exclude is not None:
        intersecting_columns = [column for column in intersecting_columns if column not in columns_to_exclude]
    
    # Compare the values in the intersecting columns
    # See here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html 
    differences = df1[intersecting_columns].compare(df2[intersecting_columns])

    # Return the number of differences
    return len(differences)

num_differences = check_for_differences_between_df(updated_annotations_reset_index, original_annotations, columns_to_exclude=["label_last_updated_at"])

# Upload the updated annotations to Google Storage and track the changes
from utils.gcp_utils import upload_to_gs, rename_blob, delete_blob
from utils.wandb_utils import wandb_add_artifact_with_reference
from utils.misc import get_now_time

UPDATED_ANNOTATIONS_TARGET_FILENAME = "updated_annotations.csv"
ORIGINAL_ANNOTATIONS_TARGET_FILENAME = "annotations.csv"

# Export the updated annotations to a CSV
columns_to_export = config.columns_to_export
print(f"[INFO] Exporting the following columns to {UPDATED_ANNOTATIONS_TARGET_FILENAME}: {columns_to_export}")

# TODO: Check if the updated_annotations_reset_index and the original_annotations actually differ, if so save them and upload them, else exit
if num_differences > 0:
    print(f"[INFO] {num_differences} changes to annotations.csv, updated label files and original annotations are different, saving the updated annotations.csv")

    # Export the updated_annotations_reset_index to a csv
    updated_annotations_reset_index[columns_to_export].to_csv(UPDATED_ANNOTATIONS_TARGET_FILENAME, index=False)

    # Upload the updated CSV to Google Storage
    upload_to_gs(bucket_name=GS_BUCKET_NAME, 
                 source_file_name=UPDATED_ANNOTATIONS_TARGET_FILENAME, 
                 destination_blob_name=UPDATED_ANNOTATIONS_TARGET_FILENAME)

    # Rename the old CSV on Google Storage
    bucket_to_move_old_annotations_to = "old_annotations"
    name_to_rename_old_annotations = os.path.join(bucket_to_move_old_annotations_to, f"{get_now_time()}_old_annotations.csv")

    rename_blob(bucket_name=GS_BUCKET_NAME,
                blob_name=ORIGINAL_ANNOTATIONS_TARGET_FILENAME,
                new_name=name_to_rename_old_annotations)

    # Rename the "updated_annotations.csv" on Google Storage to "annotations.csv" 
    rename_blob(bucket_name=GS_BUCKET_NAME,
                blob_name=UPDATED_ANNOTATIONS_TARGET_FILENAME,
                new_name=ORIGINAL_ANNOTATIONS_TARGET_FILENAME)

    # Track the changes in the annotations with Weights & Biases
    annotations_path_on_gcs = f"gs://{GS_BUCKET_NAME}/{ORIGINAL_ANNOTATIONS_TARGET_FILENAME}"
    wandb_add_artifact_with_reference(wandb_run=run,
                                      artifact_name="food_vision_labels",
                                      artifact_type="labels",
                                      description="Labels for FoodVision project",
                                      reference_path=annotations_path_on_gcs)
else:
    print("[INFO] No changes to annotations.csv, updated label files and original annotations are the same, try fixing/updating the label files via fix_labels.py and try again")

# TODO: potentiall make these args? or in the config?
# Delete and cleanup
DELETE_LOCAL_UPDATED_ANNOTATIONS_FILES = True
DELETE_GOOGLE_STORAGE_UPDATED_ANNOTATIONS_FILES = True

if DELETE_LOCAL_UPDATED_ANNOTATIONS_FILES:
    # Remove the local updated annotations files from local machine
    label_studio_export_paths = list(Path(LABELS_TO_FIX_DOWNLOAD_DIR).glob("*"))

    num_removed = 0
    for path in label_studio_export_paths:
        print(f"[INFO] Removing {path}...")
        path.unlink()
        num_removed += 1
    print(f"[INFO] Removed {num_removed} local updated annotation files.")
else:
    # Remove the local updated annotations files
    label_studio_export_paths = list(Path(LABELS_TO_FIX_DOWNLOAD_DIR).glob("*"))
    print(f"[INFO] DELETE_LOCAL_UPDATED_ANNOTATIONS_FILES is False, not deleting, number of local annotation files: {len(label_studio_export_paths)}")

blob_list_to_delete = get_list_of_blobs(bucket_name=GS_BUCKET_NAME,
                                        prefix=GS_CLASSIFICATION_LABELS_TO_FIX_PREFIX,
                                        names_only=False)

if len(blob_list_to_delete) > 0:
    if DELETE_GOOGLE_STORAGE_UPDATED_ANNOTATIONS_FILES:
        print("[INFO] Deleting updated annotations files from Google Storage...")
        num_deleted = 0
        for blob in tqdm(list(blob_list_to_delete)):
            print(f"[INFO] Deleting {blob.name}...")
            delete_blob(bucket_name=GS_BUCKET_NAME,
                        blob_name=blob.name)
            num_deleted += 1
        print(f"[INFO] Deleted {num_deleted} updated annotations files from Google Storage.")
    else:
        print(f"[INFO] DELETE_GOOGLE_STORAGE_UPDATED_ANNOTATIONS_FILES is False, not deleting, number of updated annotations files in Google Storage: {len(list(blob_list_to_delete))}")
else:
    print(f"[INFO] No updated annotations files in Google Storage to delete.")