
from pathlib import Path

import pandas as pd
import wandb

def wandb_load_artifact(wandb_run, artifact_name, artifact_type):
    artifact = wandb_run.use_artifact(artifact_name, type=artifact_type)
    artifact_dir = artifact.download()
    return artifact_dir

### Get the truth labels
def wandb_download_and_load_labels(
    wandb_run,
    wandb_labels_artifact_name,
    wandb_labels_artifact_type="labels",
    filename="annotations.csv",  # TODO: make this a changeable parameter?
    class_name_col="class_name",
    label_col="label",
):

    # Get labels from W&B Artifacts
    labels_dir = wandb_load_artifact(
        wandb_run=wandb_run,
        artifact_name=wandb_labels_artifact_name,
        artifact_type=wandb_labels_artifact_type,
    )
    print(f"[INFO] Labels directory: {labels_dir}")

    # Create labels path
    labels_path = Path(labels_dir) / filename
    print(f"[INFO] Labels path: {labels_path}")
    annotations = pd.read_csv(labels_path)

    # Create a dictionary of class_names and labels
    class_names = annotations[class_name_col].to_list()
    class_labels = annotations[label_col].to_list()
    class_dict = dict(sorted(dict(zip(class_labels, class_names)).items()))

    # Reverse class_dict keys and values
    reverse_class_dict = dict(zip(class_dict.values(), class_dict.keys()))
    print(f"[INFO] Working with: {len(class_dict)} classes")

    # Filter class_names for unique items
    class_names = sorted(list(set(class_names)))

    assert (
        len(class_names) == len(class_dict) == len(reverse_class_dict)
    ), "Length of class_names, class_dict and reverse_class_dict should be the same"

    return annotations, class_names, class_dict, reverse_class_dict, labels_path

def wandb_add_artifact_with_reference(wandb_run, artifact_name, artifact_type, description, reference_path):
    print(f"[INFO] Logging '{artifact_name}' from '{reference_path}' to Weights & Biases...")
    artifact = wandb.Artifact(name=artifact_name, 
                              type=artifact_type,
                              description=description,
                              )
    artifact.add_reference(reference_path, max_objects=1e9) # default capability to track up to 1 billion images
    wandb_run.log_artifact(artifact)