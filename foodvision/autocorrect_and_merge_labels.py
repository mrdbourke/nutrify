"""
Downloads prediction results from the cloud.

Tries to correct "most wrong" (samples which are wrong but have high prediction probability).

How?
- Encode all text-based class_names to vectors (currently with CLIP/BLIP)
- Perform image similarity matching across text-based class_names (encode image with CLIP/BLIP)
- If CLIP & BLIP similarity matches between image and text-based class_name, replace original with matching class_name

Only works on training split (test split labels are not autocorrect).
"""
import json
import os

from pathlib import Path

import pandas as pd
import wandb
import torch

from PIL import Image
from tqdm.auto import tqdm

# Import config
from configs.default_config import config

# Connect to GCP
from utils.gcp_utils import set_gcp_credentials, test_gcp_connection
set_gcp_credentials(path_to_key="utils/google-storage-key.json")
test_gcp_connection()

# Try and import LAVIS
try:
    print(f"[INFO] Attempting to import LAVIS (see: https://github.com/salesforce/LAVIS)")
    from lavis.models import load_model_and_preprocess
    print(f"[INFO] Successfully imported LAVIS.")
except:
    print(f"[ERROR] Failed to import LAVIS. Please see: https://github.com/salesforce/LAVIS, LAVIS is required for BLIP and CLIP models, exiting...")
    exit()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup variables
WANDB_PROJECT = config.wandb_project
WANDB_RUN_TAGS = ["update_and_merge_auto_labels"]
WANDB_JOB_TYPE = "merge_autocorrected_labels_from_vision_language_models"
WANDB_RUN_NOTES = "Autocorrect training labels with vision and language models (CLIP and BLIP)."

WANDB_MODEL = config.wandb_model_artifact
WANDB_DATASET = config.wandb_dataset_artifact
WANDB_LABELS = config.wandb_labels_artifact

# Annotations columns to export (target columns for labels file to be uploaded to GCP)
columns_to_export = config.annotations_columns_to_export

# Setup Weights and Biases
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

# Load predictions
WANDB_TRAIN_PREDS = config.wandb_train_preds_artifact

predictions_dir = wandb_load_artifact(wandb_run=run,
                                      artifact_name=WANDB_TRAIN_PREDS, 
                                      artifact_type="predictions")
                                
print(f"[INFO] Train predictions dir: {predictions_dir}")

# TODO: make this possible for train/test sets (could merge the CSV files?)
csv_files = list(Path(predictions_dir).glob("*.csv"))[0]
print(f"[INFO] Train predictions CSV files: {csv_files}")

# Load the csv from predictions_dir
train_preds_df = pd.read_csv(csv_files)

# Get the top_n_preds for each row
top_n_preds = train_preds_df["top_n_preds"].to_list()
top_n_preds = [eval(preds) for preds in top_n_preds]

# Create a column called top_n_pred_classes which is a list of the top_n_pred_classes
train_preds_df["top_n_pred_classes"] = [", ".join([pred["pred_class"] for pred in preds]) for preds in top_n_preds]

# Find the top 100 examples where the model was most confident but wrong
def find_n_most_wrong(df, n_most_wrong=100, true_label="true_label", pred_label="pred_label", pred_prob="pred_prob"):
    # Find the top n_most_wrong examples where the model was most confident but wrong
    df_wrong = df[df[pred_label] != df[true_label]]
    return df_wrong.sort_values(pred_prob, ascending=False).head(n_most_wrong)

# Create a dataframe of the X most wrong examples
num_to_try_and_autocorrect = config.num_to_try_and_autocorrect
print(f"[INFO] Trying to autocorrect {num_to_try_and_autocorrect} most wrong examples...")
most_wrong = find_n_most_wrong(train_preds_df, n_most_wrong=num_to_try_and_autocorrect)

# Turn most wrong to a list of dictionaries
most_wrong_dicts = most_wrong.to_dict("records")

# Get number of samples where prediction is in top-n and not in top-n
num_pred_in_top_n = len(most_wrong[most_wrong.pred_in_top_n == True])
num_pred_not_in_top_n = len(most_wrong[most_wrong.pred_in_top_n == False])

print(f"[INFO] Number of wrong samples where prediction is in top-n: {num_pred_in_top_n}/{len(most_wrong)} ({round(num_pred_in_top_n/len(most_wrong)*100, 2)}%)")
print(f"[INFO] Number of wrong samples where prediction is not in top-n: {num_pred_not_in_top_n}/{len(most_wrong)} ({round(num_pred_not_in_top_n/len(most_wrong)*100, 2)}%)")

# TODO: move to utils.py?
def open_image(image_path_or_PIL):
    try:
        return Image.open(image_path_or_PIL).convert("RGB")
    except:
        return image_path_or_PIL.convert("RGB")

# Start Vision-Language modelling section
blip_feature_extractor_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", 
                                                                                         model_type="base", 
                                                                                         is_eval=True, 
                                                                                         device=device)

def blip_get_image_features(image,
                            model=blip_feature_extractor_model, 
                            vis_processors=vis_processors, 
                            device=device,
                            low_dim=True):

    # Make sure model and image are on the same device
    blip_feature_extractor_model.to(device)
                 
    # Preprocess image
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    # Turn image into sample dict
    sample = {"image": image}

    # Get features
    features_image = model.extract_features(sample, mode="image")

    # Turn features into low-dim
    if low_dim:
        features_image = features_image.image_embeds_proj
    else:
        features_image = features_image.image_embeds

    return features_image

def blip_get_text_features(text,
                           model=blip_feature_extractor_model, 
                           txt_processors=txt_processors, 
                           device=device,
                           low_dim=True):
                 
    # Preprocess image
    text_input = txt_processors["eval"](text)

    # Turn image into sample dict
    sample = {"text_input": [text_input]}

    # Get features
    features_text = model.extract_features(sample, mode="text")

    # Turn features into low-dim
    if low_dim:
        features_text = features_text.text_embeds_proj
    else:
        features_text = features_text.text_embeds

    return features_text.to(device)

### Load CLIP model and create helper functions
clip_feature_extractor_model, clip_vis_processors, clip_txt_processors = load_model_and_preprocess(name="clip_feature_extractor", 
                                                                                                   model_type="ViT-B-16", 
                                                                                                   is_eval=True, 
                                                                                                   device=device)

# Create a function to get CLIP image features
def clip_get_image_features(image, 
                            clip_feature_extractor_model, 
                            clip_vis_processors, 
                            device=device):
    processed_image = clip_vis_processors["eval"](image).unsqueeze(0).to(device)
    sample = {"image": processed_image}
    clip_image_features = clip_feature_extractor_model.extract_features(sample)
    # clip_image_features = clip_features.image_embeds_proj
    return clip_image_features

def clip_get_text_features(text, clip_txt_processors, clip_feature_extractor_model, device=device):
    text_input = clip_txt_processors["eval"](text)
    sample = {"text_input": [text_input]}
    features_text = clip_feature_extractor_model.extract_features(sample)
    return features_text

### Calculate class_names text embeddings
print(f"[INFO] Calculating class_names text embeddings with CLIP & BLIP...")

# Get BLIP features a list of all class names
def blip_get_text_features_of_list(target_list: list, device=device):
    blip_class_name_features = []
    for class_name in tqdm(target_list, desc="Calculating BLIP text features for class names"):
        class_name_features = blip_get_text_features(text=class_name, low_dim=True)
        # print(class_name_features[-1][-1].shape)
        last_dim_of_class_name_features = class_name_features[-1][0].unsqueeze(0)
        blip_class_name_features.append(last_dim_of_class_name_features.to(device))
    
    return blip_class_name_features

# Get BLIP features for a list of all class names
blip_class_name_features = blip_get_text_features_of_list(target_list=class_names, device="cpu")

# Get CLIP features for a list of all class names
def clip_get_text_features_of_list(target_list: list, device=device):

    clip_class_name_features = []
    from tqdm.auto import tqdm
    for class_name in tqdm(target_list, desc="Calculating CLIP text features for class names"):
        class_name_features = clip_get_text_features(class_name, clip_txt_processors, clip_feature_extractor_model)
        clip_class_name_features.append(class_name_features.to(device))

    return clip_class_name_features

# Get CLIP features for a list of all class names
clip_class_name_features = clip_get_text_features_of_list(target_list=class_names, device="cpu")

# Create helper function for make similarity dict
def get_sorted_similarity_dict(image_features,
                               class_name_features: list,
                               class_names: list,
                               device=device):
    """Gets a sorted similarity dict between image features and text features."""
    # class_name_features = class_names embedded in feature space
    # class_names = list of class names (e.g. ["pizza", "hot dog", "hamburger", ...])

    # Get similarities between image features and text features
    similarites = [image_features.to(device).detach() @ text_features.t().to(device).detach() for text_features in list(class_name_features)]

    # Map similarities to class names
    similarity_dict = {class_name: similarity[0][0] for class_name, similarity in zip(class_names, similarites)}

    # Sort similarity dict by values
    similarity_dict_sorted = {k: v for k, v in sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)}

    return similarity_dict_sorted

### Perform similarity matching across all images and class names
import random 
samples_to_sample = random.sample(most_wrong_dicts, len(most_wrong_dicts))
image_auto_correction_dict_list = []
print(f"[INFO] Performing similarity matching across all images and class names...")
for sample in tqdm(samples_to_sample, desc="Similarity matching image(s) @ text(s)"):
    image_auto_correction_dict = {}

    # Open the image
    image_path = sample["image_path"]
    image = Image.open(image_path).convert("RGB")

    # Get the image name
    image_name = image_path.split("/")[-1]

    # print(f"True: class: {sample['true_class']}")
    # print(f"Pred class: {sample['pred_class']}")
    # print(f"Top-n preds: {sample['top_n_pred_classes']}")
    
    ### CLIP
    # Get CLIP image features
    # TODO: in the future I could make it so the image features are calculated and cached to prevent having to calculate them on the fly...
    clip_img_features = clip_get_image_features(image, 
                                                clip_feature_extractor_model,
                                                clip_vis_processors,
                                                device="cuda") # perform image feature extraction on GPU (faster than CPU)

    # Get CLIP similarity dict
    clip_similarity_dict_sorted = get_sorted_similarity_dict(clip_img_features,
                                                             clip_class_name_features,
                                                             class_names,
                                                             device="cpu") # perform similarity matching on CPU (to prevent memory errors on GPU with large numbers of classes)

    # # Print top-5 clip_similarity_dict_sorted
    # print("\nCLIP top-5:")
    # print(list(clip_similarity_dict_sorted.items())[:5])

    # Get CLIP similarity dict top 5
    clip_similarity_dict_sorted_top_5 = list(clip_similarity_dict_sorted.items())[:5]
    clip_similiarty_dict_sorted_top_1_class_name = clip_similarity_dict_sorted_top_5[0][0] # just get class name
    
    ### BLIP
    # Get BLIP image features
    blip_img_features = blip_get_image_features(image, device="cuda")[-1][0].unsqueeze(0) # original shape: [1, 197, 256] -> [1, 256] (remove middle dimension)

    # Get BLIP similarity dict
    blip_similarity_dict_sorted = get_sorted_similarity_dict(blip_img_features,
                                                             blip_class_name_features,
                                                             class_names,
                                                             device="cpu") # perform similarity matching on CPU (to prevent memory errors on GPU with large numbers of classes)
    
    # Get BLIP similarity dict top 5
    blip_similarity_dict_sorted_top_5 = list(blip_similarity_dict_sorted.items())[:5]
    blip_similiarty_dict_sorted_top_1_class_name = blip_similarity_dict_sorted_top_5[0][0] # just get class name

    # # Print top-5 blip_similarity_dict_sorted
    # print("\nBLIP top-5:")
    # print(list(blip_similarity_dict_sorted.items())[:5])
    
    # Show the image 
    # plt.imshow(image)

    # Add details to image_auto_correction_dict
    image_auto_correction_dict["image_path"] = image_path
    image_auto_correction_dict["image_name"] = image_name
    image_auto_correction_dict["true_class"] = sample["true_class"]
    image_auto_correction_dict["pred_class"] = sample["pred_class"]
    image_auto_correction_dict["top_n_pred_classes"] = sample["top_n_pred_classes"]
    image_auto_correction_dict["pred_in_top_n"] = sample["pred_in_top_n"]
    image_auto_correction_dict["clip_similarity_dict_sorted_top_5"] = clip_similarity_dict_sorted_top_5
    image_auto_correction_dict["clip_similiarty_dict_sorted_top_1_class_name"] = clip_similiarty_dict_sorted_top_1_class_name
    image_auto_correction_dict["blip_similarity_dict_sorted_top_5"] = blip_similarity_dict_sorted_top_5
    image_auto_correction_dict["blip_similiarty_dict_sorted_top_1_class_name"] = blip_similiarty_dict_sorted_top_1_class_name

    # Append to list
    image_auto_correction_dict_list.append(image_auto_correction_dict)  

# Turn list into dataframe
image_auto_correction_df = pd.DataFrame(image_auto_correction_dict_list)

# Find samples where CLIP and BLIP top_1_class_names are the same
image_auto_correction_df["clip_blip_top_1_class_names_match"] = image_auto_correction_df["clip_similiarty_dict_sorted_top_1_class_name"] == image_auto_correction_df["blip_similiarty_dict_sorted_top_1_class_name"]
num_clip_blip_match = image_auto_correction_df["clip_blip_top_1_class_names_match"].value_counts()[True]

print(f"[INFO] Number of samples where CLIP and BLIP top-1 match: {num_clip_blip_match}/{len(image_auto_correction_df)} ({num_clip_blip_match/len(image_auto_correction_df)*100:.2f}%)")

# Find samples where CLIP and BLIP top_1_class_names are the same but the CLIP/BLIP predictions are different to the true class
image_auto_correction_df["clip_blip_top_1_class_names_match_different_to_true_class"] = (image_auto_correction_df["clip_blip_top_1_class_names_match"] == True) & (image_auto_correction_df["clip_similiarty_dict_sorted_top_1_class_name"] != image_auto_correction_df["true_class"])
num_clip_blip_match_different_to_true_class = image_auto_correction_df["clip_blip_top_1_class_names_match_different_to_true_class"].value_counts()[True]

print(f"[INFO] Number of samples where CLIP and BLIP top-1 match but are different to the true class: {num_clip_blip_match_different_to_true_class}/{len(image_auto_correction_df)} ({num_clip_blip_match_different_to_true_class/len(image_auto_correction_df)*100:.2f}%)")

# Create a view of top-1 match between CLIP/BLIP but different to true class
clip_blip_top_1_class_names_match_different_to_true_class = image_auto_correction_df[image_auto_correction_df["clip_blip_top_1_class_names_match_different_to_true_class"] == True]
num_rows_where_clip_blip_top_1_match_but_different_to_true = len(clip_blip_top_1_class_names_match_different_to_true_class)
print(f"Number of rows where CLIP and BLIP top_1 class names match but are different to the true class: {num_rows_where_clip_blip_top_1_match_but_different_to_true}")

# Make a copy for easy editing
clip_blip_match = clip_blip_top_1_class_names_match_different_to_true_class.copy()

# Update annotations with CLIP/BLIP top-1 class names
clip_blip_match["updated_label"] = clip_blip_match["clip_similiarty_dict_sorted_top_1_class_name"]

clip_blip_match_updated_labels = clip_blip_match[["image_name", "updated_label"]]

### Start to update annotations

# Update the annotations dataframe with the new labels
updated_annotations = pd.merge(original_annotations, clip_blip_match_updated_labels, how="left", on="image_name")

# Set the updated_annotations "class_name" column to the value of the "updated_label" column if it's not null
import numpy as np 
updated_annotations["class_name"] = np.where(updated_annotations["updated_label"].isnull(), updated_annotations["class_name"], updated_annotations["updated_label"])

# Map the class_name to label in updated_annotations with reverse_class_dict
updated_annotations["label"] = updated_annotations["class_name"].map(reverse_class_dict)

# Get the column names where updated_annotations and original_annotations are the same
intersecting_columns = list(set(updated_annotations.columns).intersection(set(original_annotations.columns)))

# Compare the updated_annotations and original_annotations
try:
    changed_rows = updated_annotations[intersecting_columns].compare(original_annotations[intersecting_columns], result_names=("updated_annotations", "original_annotations")) # result_names requires pandas v1.5+
except:
    changed_rows = updated_annotations[intersecting_columns].compare(original_annotations[intersecting_columns])

### Add label source and time of update
updated_indexes = updated_annotations[intersecting_columns].compare(original_annotations[intersecting_columns])

# Set the label_source column to "auto_corrected" for the updated indexes
updated_annotations.loc[updated_indexes.index, "label_source"] = "auto_labelled_clip_and_blip_match"

# Set the last_updated column to the current date for the updated indexes
import datetime
updated_annotations.loc[updated_indexes.index, "label_last_updated_at"] = datetime.datetime.now()

# TODO: move to utils?
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

# Check for differences between updated_annotations and original_annotations
num_differences = check_for_differences_between_df(updated_annotations, 
                                                   original_annotations, 
                                                   columns_to_exclude=["label_last_updated_at", "label_source"])

assert num_differences == len(clip_blip_match_updated_labels), "The number of rows in changed_rows should be the same as the number of rows in clip_blip_match_updated_labels"

### Upload and save annotations
GS_BUCKET_NAME = config.gs_bucket_name
UPDATED_ANNOTATIONS_TARGET_FILENAME = "updated_annotations.csv"
ORIGINAL_ANNOTATIONS_TARGET_FILENAME = "annotations.csv"

# TODO: Check if the updated_annotations_reset_index and the original_annotations actually differ, if so save them and upload them, else exit
from utils.gcp_utils import upload_to_gs, rename_blob
from utils.wandb_utils import wandb_add_artifact_with_reference
from utils.misc import get_now_time

if num_differences > 0:
    print(f"[INFO] {num_differences} changes to annotations.csv, updated label files and original annotations are different, saving the updated annotations.csv file and uploading it to Google Storage...")

    # Export the updated_annotations to a csv
    print(f"[INFO] Exporting columns: {columns_to_export}...")
    updated_annotations[columns_to_export].to_csv(UPDATED_ANNOTATIONS_TARGET_FILENAME, index=False)

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

else:
    print("[INFO] No changes to annotations.csv, updated label files and original annotations are the same, try fixing/updating the label files and try again")

# TODO: move this into another script? 
# TODO: make it easier to track Artifact changes
# TODO: e.g. there's a dedicated Artifact tracker file that gets run after any changes
# Always track changes to W&B (this should automatically detect if there is/isn't changes and track)
annotations_path_on_gcs = f"gs://{GS_BUCKET_NAME}/{ORIGINAL_ANNOTATIONS_TARGET_FILENAME}"
wandb_add_artifact_with_reference(wandb_run=run,
                                  artifact_name="food_vision_labels",
                                  artifact_type="labels",
                                  description="Labels for FoodVision project",
                                  reference_path=annotations_path_on_gcs)
