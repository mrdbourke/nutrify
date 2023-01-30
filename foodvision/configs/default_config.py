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

# Paths
config.path_to_gcp_credentials = "foodvision/utils/google-storage-key.json" # TODO: make sure this path exists, else error
config.gs_bucket_name = "food_vision_bucket_with_object_versioning"
config.gs_image_storage_path = "https://storage.cloud.google.com/food_vision_bucket_with_object_versioning/all_images/"

# Weights and Biases
config.wandb_project = "test_wandb_artifacts_by_reference"
config.wandb_job_type = "predict with trained food vision model"
config.wandb_run_tags = ["train"]
config.wandb_run_notes = "" 

config.wandb_model_artifact = "trained_model:latest"
config.wandb_dataset_artifact = "food_vision_199_classes_images:latest"
config.wandb_labels_artifact = "food_vision_labels:latest"

config.wandb_train_preds_artifact = "train_predictions:latest"

# Dataset
config.workers = 4
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

if __name__ == "__main__":
    print(config)