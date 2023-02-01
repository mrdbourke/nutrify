"""
Script to evaluate FoodVision models on train/test sets and save predictions
to Weights & Biases and Google Storage.
"""
import argparse
import os
import random
import torch
import timm
import wandb
import yaml

import pandas as pd

from data_loader import FoodVisionReader
from pathlib import Path
from timm.models import create_model
from timm.data import create_transform, ImageDataset
from tqdm.auto import tqdm
from PIL import Image
from google.cloud import storage

from sklearn.metrics import accuracy_score, classification_report

# Create config parser (to get baseline config parameters)
config_parser = parser = argparse.ArgumentParser(description="Training config file")
parser.add_argument("-c", "--config", default="configs.default_config", type=str, help="config file path (default: configs.default_config)")

# Create arguments
parser = argparse.ArgumentParser(description="Evaluate a FoodVision model.")
group = parser.add_argument_group("Dataset parameters")
group.add_argument("--dataset", "-d", default="", type=str, help="dataset to use")
group.add_argument(
    "--train-split", default="train", help="dataset train split (defailt: train)"
)
group.add_argument(
    "--test-split", default="test", help="dataset test split (defailt: test)"
)
group.add_argument(
    "--class-map", default="", help="path to class to idx mapping file (default: '')"
)
group.add_argument(
    "--workers",
    default=16,
    type=int,
    help="number of workers for dataloader (default :4)",
)
group.add_argument(
    "--image-size",
    default=224,
    type=int,
    help="image size to resize images to (default: 224)",
)
group.add_argument(
    "--num-most-wrong",
    default=100,
    type=int,
    help="number of most wrong predictions to log (default: 100)",
)
group.add_argument(
    "--num-top-n-preds",
    default=5,
    type=int,
    help="number of top_n preds to log and evaluate with (default: 5)",
)
group.add_argument(
    "--num-samples-to-predict-on",
    default="all",
    help="number of samples to predict on, can also be int, e.g. 100 (default: 'all')",
)
group.add_argument(
    "--pred-splits-to-predict-on",
    default=["train", "test"],
    type=list,
    help="splits to predict on, e.g. ['train', 'test'] (default: ['train', 'test'])",
)

# Create model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model",
    default="coatnext_nano_rw_224",
    help="model to use (default: 'coatnext_nana_rw_224')",
)
group.add_argument(
    "--pretrained", default=True, help="use pretrained weights (default: True)"
)

# Create Weights and Biases parameters
# TODO: make it so you can use W&B if you want to but don't have to if you dont want to... (this can come later, simple first)
group = parser.add_argument_group("Weights and Biases parameters")
group.add_argument(
    "--wandb-project",
    default="test_wandb_artifacts_by_reference",
    help="Weights & Biases project name, e.g. 'food_vision' (default: 'test_wandb_artifacts_by_reference')",
)
group.add_argument(
    "--wandb-job-type",
    default="predict with trained food vision model",
    help="Weights & Biases job type, e.g. 'predict with trained food vision model' (default: 'predict with trained food vision model')",
)
group.add_argument(
    "--wandb-dataset-artifact",
    default="food_vision_199_classes_images:latest",
    help="Weights & Biases dataset artifact name",
)
group.add_argument(
    "--wandb-labels-artifact",
    default="food_vision_labels:latest",
    help="Weights & Biases labels artifact name",
)
group.add_argument(
    "--wandb-models-artifact",
    default="trained_model:latest",
    help="Weights & Biases trained model artifact name",
)
group.add_argument(
    "--wandb-run-tags",
    default=["predict_and_evaluate"],
    type=list,
    help="Weights & Biases run tags, for example (['predict_and_evaluate'])",
)
group.add_argument(
    "--wandb-run-notes",
    "-wb_notes",
    default="",
    help="Weights & Biases run notes, similar to writing a git commit message, 'what did you do?' (default '')",
)

# Misc parameters
group = parser.add_argument_group("Misc parameters")
group.add_argument(
    "--quick-experiment",
    "-qe",
    default=False,
    type=bool,
    help="whether to run a full run-through quick experiment (limits samples to first 100 units only) (default: False)",
)
group.add_argument(
    "--predictions-out-dir",
    default="predictions",
    type=str,
    help="path to model output directory (default: 'predictions')",
)

# See: https://github.com/rwightman/pytorch-image-models/blob/3aa31f537d5fbf6be8f1aaf5a36f6bbb4a55a726/train.py#L352
# See here: https://docs.python.org/3/library/argparse.html#partial-parsing 
def _parse_args():
    """Parses command line arguments.

    Default behaviour:
    - take in config file (e.g. configs.default_config.py) and use those as defaults
    - take in command line arguments and override defaults if necessary

    See:
    - https://github.com/rwightman/pytorch-image-models/blob/3aa31f537d5fbf6be8f1aaf5a36f6bbb4a55a726/train.py#L352
    - https://docs.python.org/3/library/argparse.html#partial-parsing 
    """
    config_args, remaining = config_parser.parse_known_args()

    # Parse config file
    if config_args.config:
        from importlib import import_module
        # See here: https://stackoverflow.com/a/67692/12434862
        # This is equivalent to: from configs.default_config import config
        config_module = getattr(import_module(config_args.config), "config")
        # print("\n#### Config module:\n")
        # print(config_module)
        parser.set_defaults(**config_module.__dict__)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__)

    return args, args_text

# TODO: perhaps the args could populate a config and then the config is used throughout the script?
args, args_text = _parse_args()

# Connect to GCP
from utils import set_gcp_credentials, test_gcp_connection
set_gcp_credentials(path_to_key=args.path_to_gcp_credentials)
test_gcp_connection()

GS_BUCKET = args.gs_bucket_name

# Set devices
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Set seeds
from utils import seed_everything
seed_everything(args.seed)

# Setup Weights & Biases
run = wandb.init(
    project=args.wandb_project,
    job_type=args.wandb_job_type,
    tags=["predict_and_evaluate"],
    notes=args.wandb_run_notes,
)

# Add args config to Weights & Biases
wandb.config.update(args)

# Download dataset artifact
from utils.wandb_utils import wandb_load_artifact
images_dir = wandb_load_artifact(
    wandb_run=run,
    artifact_name=args.wandb_dataset_artifact,
    artifact_type="dataset")

# Download labels artifact
from utils.wandb_utils import wandb_download_and_load_labels

annotations, class_names, class_dict, reverse_class_dict, labels_path = wandb_download_and_load_labels(wandb_run=run, 
                                                                                                            wandb_labels_artifact_name=args.wandb_labels_artifact)

# Download model artifact
model_at_dir = wandb_load_artifact(wandb_run=run,
                                   artifact_name=args.wandb_models_artifact,
                                   artifact_type="model")
print(f"[INFO] Model artifact directory: {model_at_dir}")

# Get list of all files with "*.pth" in model_at_dir
model_path = str(list(Path(model_at_dir).rglob("*.pth"))[0])
print(f"[INFO] Model path: {model_path}")

# Create the model with timm
model = timm.create_model(
    model_name=args.model, pretrained=args.pretrained, num_classes=len(class_dict)
)
model.load_state_dict(torch.load(model_path))

# Create the data transform
transform = create_transform(input_size=args.image_size, is_training=False)

# Add predictions to Google Storage
from utils import upload_to_gs


def pred_on_image(
    model,
    class_dict,
    image_path,
    transform=transform,
    device=device,
    num_top_n_preds=args.num_top_n_preds,
):
    """Makes a predition on a single image.

    Args:
        model (_type_): A trained PyTorch model.
        class_dict (Dict): Mapping of class labels to class names.
        image_path (str): String path to target image.
        transform (_type_): Transform to perform on image before being passed to model.
        device (torch.device): Device to make prediction on (e.g. "cuda" or "cpu").
        num_top_n_preds (int, optional): Number of top_n predictions to return. Defaults to 5.

    Returns:
        dict: a dictionary containing the image path, the top n predictions, and the top n prediction probabilities.
        For example:

            pred_dict = {
                "image_path": image_path,
                "pred_label": pred_label,
                "pred_prob": pred_prob.max().item(),
                "pred_class": pred_class,
                "top_n_preds": top_n_pred_dict}
    """

    # Open image
    img = Image.open(image_path)

    # Make sure model on target device
    model.to(device)

    # Turn on model eval mode and make prediction
    model.eval()
    with torch.inference_mode():
        img_tensor = transform(img).unsqueeze(0).to(device)
        pred = model(img_tensor)
        pred_prob = torch.softmax(pred, dim=1)
        pred_label = pred_prob.argmax().item()
        pred_class = class_dict[pred_label]

    # Get top_n predictions
    top_n_pred_tensor = pred_prob.topk(num_top_n_preds)
    top_n_pred_probs = top_n_pred_tensor.values.tolist()[0]
    top_n_pred_labels = top_n_pred_tensor.indices.tolist()[0]
    top_n_pred_classes = [class_dict[i] for i in top_n_pred_labels]

    # Zip together top_n_pred_labels and top_n_pred_classes
    top_n_pred_dict = [
        {"pred_prob": pred_prob, "pred_label": pred_label, "pred_class": pred_class}
        for pred_prob, pred_label, pred_class in zip(
            top_n_pred_probs, top_n_pred_labels, top_n_pred_classes
        )
    ]

    # Create pred dict
    pred_dict = {
        "image_path": image_path,
        "pred_label": pred_label,
        "pred_prob": pred_prob.max().item(),
        "pred_class": pred_class,
        "top_n_preds": top_n_pred_dict,
    }

    return pred_dict


# TODO: Log the predictions to Weights & Biases/Google Storage
# TODO: check the Artifacts and see if they can just be stored as reference rather than
# always being uploaded straight to W&B and Google Storage
def log_predictions_artifact_to_wandb(
    predictions_path,
    description="Predictions file containing prediction results and paths to data.",
    project=args.wandb_project,
    run=run,
):
    """Logs a predictions file to Weights & Biases as an artifact."""
    print(f"[INFO] Logging {predictions_path} to Weights & Biases...")

    # Create a wandb artifact
    pred_artifact = wandb.Artifact(
        name=predictions_path.stem,
        type="predictions",
        description=description,
    )

    # Add the model file to the artifact
    pred_artifact.add_file(predictions_path)

    # Save the model to W&B
    wandb.save(str(predictions_path))

    # TODO: Add reference file to GCP
    # model_artifact.add_reference(model_gcs_path)

    # Log the artifact to W&B
    # run = wandb.init(project=project)
    run.log_artifact(pred_artifact)


# Make function to log most wrong predictions to Weights & Biases
def log_most_wrong_to_wandb(most_wrong_df, name, run=run):
    """Logs the most wrong predictions to Weights & Biases Tables."""
    print(
        f"[INFO] Logging top {len(most_wrong_df)} most wrong predictions to Weights & Biases Tables."
    )

    # Turn most_wrong into list of dictionaries
    most_wrong_dict = most_wrong_df.to_dict(orient="records")

    # Turn the dict images into wandb Image objects
    for sample in most_wrong_dict:
        # Convert image path to str
        sample["image_path"] = str(sample["image_path"])
        sample["image"] = wandb.Image(str(sample["image_path"]))

    # Get column names
    columns = list(most_wrong_dict[0].keys())

    # Log the most wrong predictions to Weights & Biases Tables
    wandb_table = wandb.Table(columns=columns)

    for sample in most_wrong_dict:
        wandb_table.add_data(*[sample[column] for column in columns])

    # Save the table
    run.log({name: wandb_table})


# Make a function to analyze a predictions dataframe and output metrics
def analyze_predictions_dataframe(df, split="train"):
    # Make a column called correct if pred_label matches true_label
    df["pred_correct"] = df["pred_label"] == df["true_label"]

    # Make a column called top_n to see if the true_label is in the top_n_preds
    df["pred_in_top_n"] = df.apply(
        lambda x: x["true_label"] in [i["pred_label"] for i in x["top_n_preds"]], axis=1
    )

    # Calculate metrics on the predictions and save them to Weights & Biases
    top_1_acc = accuracy_score(df["true_label"], df["pred_label"])
    top_5_acc = accuracy_score(df["pred_in_top_n"], [True] * len(df))

    # Calculate scikit-learn classification_report for df
    # Get unique classes and predictions (this is required for when the number of samples being predicted on doesn't contain all the classes)
    # For example, you predict on 100 random samples but there are ~200 classes, there's not full coverage.
    # For a sufficiently high number of predictions, this won't be required.
    pred_labels_unique = list(df.pred_label.unique())
    true_labels_unique = list(df.true_label.unique())
    all_labels_unique = list(set(pred_labels_unique + true_labels_unique))
    all_labels_unique_as_classes = [class_dict[i] for i in all_labels_unique]

    # Create classification report as dictionary (this will give precision, recall and F1-score for each class)
    classification_dict = classification_report(
        df["true_label"],
        df["pred_label"],
        target_names=all_labels_unique_as_classes,
        zero_division=0,
        output_dict=True,
    )

    # Separate out the "accuracy", "weighted avg", "macro avg" rows
    classification_dict.pop("accuracy")
    weighted_avg = classification_dict.pop("weighted avg")
    macro_avg = classification_dict.pop("macro avg")

    # Turn the classification_dict into a DataFrame
    classification_report_df = pd.DataFrame(classification_dict).T

    # Turn the index of classification_df into a column
    classification_report_df.reset_index(inplace=True)

    # Rename the index column to "class_name"
    classification_report_df.rename(columns={"index": "class_name"}, inplace=True)

    # Create a dictionary of metrics
    metrics_dict = {
        f"{split}_top_1_acc": top_1_acc,
        f"{split}_top_5_acc": top_5_acc,
        f"{split}_weighted_avg_precision": weighted_avg["precision"],
        f"{split}_weighted_avg_recall": weighted_avg["recall"],
        f"{split}_weighted_avg_f1-score": weighted_avg["f1-score"],
        f"{split}_macro_avg_precision": macro_avg["precision"],
        f"{split}_macro_avg_recall": macro_avg["recall"],
        f"{split}_macro_avg_f1-score": macro_avg["f1-score"],
    }

    return df, metrics_dict, classification_report_df


# Make predictions and save them
pred_splits_to_predict_on = args.pred_splits_to_predict_on
num_samples = args.num_samples_to_predict_on
num_most_wrong = args.num_most_wrong
num_top_n_preds = args.num_top_n_preds

# Create empty list to save whole lists of prediction dictionaries too
pred_dicts = []
for pred_split_to_predict_on in pred_splits_to_predict_on:
    # Create empty list to save individual prediction dictionaries too
    pred_list = []
    print(f"[INFO] Predicting on {pred_split_to_predict_on} split...")

    # Create food vision reader
    food_vision_reader = FoodVisionReader(
        image_root=images_dir,
        label_root=labels_path,
        class_to_idx=class_dict,
        quick_experiment=args.quick_experiment,
        split=pred_split_to_predict_on
    )

    # Make predictions on all images (default) or a random sample of images
    if num_samples == "all" or num_samples > len(food_vision_reader):
        print(
            f"[INFO] Predicting on all {pred_split_to_predict_on} samples ({len(food_vision_reader)})..."
        )
        random_sample_idxs = random.sample(
            range(0, len(food_vision_reader)), len(food_vision_reader)
        )
    else:
        print(
            f"[INFO] Predicting on {num_samples} random {pred_split_to_predict_on} samples..."
        )
        random_sample_idxs = random.sample(
            range(0, len(food_vision_reader)), num_samples
        )

    # Loop through samples and make predictions, then save predictions to dictionary
    # for i in track(random_sample_idxs, description="Making predictions..."):
    for i in tqdm(random_sample_idxs, desc="Making predictions...", total=len(random_sample_idxs)):
        img, label = food_vision_reader[i]

        filename = Path(food_vision_reader._filename(index=i))
        image_name = filename.name
        pred_dict = pred_on_image(
            model=model,
            class_dict=class_dict,
            image_path=filename,
            transform=transform,
            device=device,
            num_top_n_preds=num_top_n_preds,
        )
        pred_dict["image_name"] = image_name
        pred_dict["true_label"] = label
        pred_dict["true_class"] = class_dict[label]
        pred_dict["split"] = pred_split_to_predict_on
        # Append invididual prediction dictionary to pred_list
        pred_list.append(pred_dict)

    # Add pred_list to pred_dicts (for later inspection)
    pred_holder = {pred_split_to_predict_on: pred_list}
    pred_dicts.append(pred_holder)

    # Turn predictions to DataFrame
    df = pd.DataFrame(pred_list)

    # Analyze the predictions and add columns to the DataFrame
    df, metrics_dict, classification_report_df = analyze_predictions_dataframe(
        df=df, split=pred_split_to_predict_on
    )

    # Log the mertics and classification report to Weights & Biases
    wandb.log(metrics_dict)
    wandb.log(
        {
            f"{pred_split_to_predict_on}_classification_report_df": wandb.Table(
                dataframe=classification_report_df
            )
        }
    )

    # Make a new directory called predictions with pathlib.Path
    predictions_dir = Path(args.predictions_out_dir)
    predictions_dir.mkdir(exist_ok=True)

    # Create predictions path
    predictions_save_path = Path(
        predictions_dir, f"{pred_split_to_predict_on}_predictions.csv"
    )
    print(
        f"[INFO] Saving {pred_split_to_predict_on} predictions to {predictions_save_path}."
    )

    # Save df to csv in folder called predictions
    df.to_csv(predictions_save_path, index=False)

    # Save predictions to Weights & Biases Artifacts
    log_predictions_artifact_to_wandb(
        predictions_save_path
    )  # TODO: could I use a GCP reference here instead of uploading the file to W&B?

    # Save predictions to GCS
    preds_gcs_path = upload_to_gs(
        bucket_name=GS_BUCKET,
        source_file_name=predictions_save_path,
        destination_blob_name=f"predictions/{predictions_save_path.name}",
    )

    # Save predictions path to Weights & Biases
    wandb.config.update(
        {f"{pred_split_to_predict_on}_predictions_path": preds_gcs_path}
    )

    # Find the 100 most wrong images
    most_wrong_df = (
        df[df["pred_label"] != df["true_label"]]
        .sort_values("pred_prob", ascending=False)
        .head(num_most_wrong)
    )

    # Save most wrong to CSV
    most_wrong_save_path = Path(
        predictions_dir,
        f"{pred_split_to_predict_on}_{num_most_wrong}_most_wrong_predictions.csv",
    )
    most_wrong_df.to_csv(most_wrong_save_path, index=False)

    # Log most wrong predictions to Weights & Biases Tables
    log_most_wrong_to_wandb(
        most_wrong_df,
        name=f"{pred_split_to_predict_on}_{num_most_wrong}_most_wrong_predictions",
    )

    # Save top_n most wrong to GCS
    most_wrong_gcs_path = upload_to_gs(
        bucket_name=GS_BUCKET,
        source_file_name=most_wrong_save_path,
        destination_blob_name=f"predictions/{most_wrong_save_path.name}",
    )

    # Save most wrong GCS path to Weights & Biases
    wandb.config.update(
        {
            f"{pred_split_to_predict_on}_{num_most_wrong}_most_wrong_predictions_path": most_wrong_gcs_path
        }
    )

# Finish the run
run.finish()
