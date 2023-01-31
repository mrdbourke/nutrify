"""
Default configuration for foodvision classification models.

Inspirations:
- https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/main/configs/default_config.py
- https://twitter.com/i/web/status/1531155343505489920
- https://github.com/mrdbourke/nutrify/issues/49
"""
import os
from types import SimpleNamespace

# Define the default configuration
config = SimpleNamespace(**{})

# Keys
config.path_to_label_studio_api_key = "utils/label_studio_api_key.json"
config.path_to_gcp_credentials = "utils/google-storage-key.json" # TODO: make sure this path exists, else error

# Paths
config.gs_bucket_name = "food_vision_bucket_with_object_versioning"
config.gs_image_storage_path = "https://storage.cloud.google.com/food_vision_bucket_with_object_versioning/all_images/"

# Weights and Biases
config.wandb_project = "test_wandb_artifacts_by_reference"
config.wandb_job_type = ""
config.wandb_run_notes = "" 
config.wandb_run_tags = [""] # NOTE: perhaps this can be per script? so the default doesn't just always end up as "train"

config.wandb_model_artifact = "trained_model:latest"
config.wandb_dataset_artifact = "food_vision_199_classes_images:latest"
config.wandb_labels_artifact = "food_vision_labels:latest"

config.wandb_train_preds_artifact = "train_predictions:latest"

# Data loading and training
config.workers = 16
config.input_size = 224
config.auto_augment = True

# Model
config.model = "coatnext_nano_rw_224"
config.pretrained = True

# Training
config.batch_size = 128
config.epochs = 10
config.label_smoothing = 0.1
config.learning_rate = 0.001
config.use_mixed_precision = True

# Misc
config.seed = 42

# Autocorrecting (only works on training split)
config.num_to_try_and_autocorrect = 1000

# Data labelling
config.annotations_columns_to_export = ["filename", 
                                        "image_name", 
                                        "class_name", 
                                        "label", 
                                        "split", 
                                        "clear_or_confusing", 
                                        "whole_food_or_dish", 
                                        "one_food_or_multiple", 
                                        "label_last_updated_at",
                                        "label_source", 
                                        "image_source"]

if __name__ == "__main__":
    print(config)