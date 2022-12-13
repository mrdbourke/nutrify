"""
Training script for FoodVision models.

TODO: adapt the script to use more elegant features such as in: 
* Timm training script: https://github.com/rwightman/pytorch-image-models/blob/main/train.py 
* Timm training script guide: 
* Potentially could put the model into a Hugging Face Transformer and then use the model
    with the Transformers class?
"""
import argparse
import json
import os
import wandb
import yaml

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader


from pathlib import Path
from typing import Dict, List
from contextlib import suppress  # suppresses amp_autocast if it's not available

# Setup Google Storage Bucket
GS_BUCKET = "food_vision_bucket_with_object_versioning"

# Setup GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-storage-key.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make it so you can import parameters with YAML if needed?

# Create arguments
parser = argparse.ArgumentParser(description="Train a FoodVision model.")

# Create dataset parameters
group = parser.add_argument_group("Dataset parameters")
group.add_argument("--dataset", "-d", default="", type=str, help="dataset to use")
group.add_argument(
    "--train_split", default="train", help="dataset train split (defailt: train)"
)
group.add_argument(
    "--test_split", default="test", help="dataset test split (defailt: test)"
)
group.add_argument(
    "--class_map", default="", help="path to class to idx mapping file (default: '')"
)
group.add_argument(
    "--workers",
    default=4,
    type=int,
    help="number of workers for dataloader (default :4)",
)
group.add_argument(
    "--image_size",
    default=224,
    type=int,
    help="image size to resize images to (default: 224)",
)

# Create model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model",
    default="coatnext_nano_rw_224",
    help="model to use (default: 'coatnext_nana_rw_224'",
)
group.add_argument(
    "--pretrained", default=True, help="use pretrained weights (default: True)"
)
# TODO: add num_classes? as arg

# Create training parameters
# TODO: creating training parameter for mixed precision training
group.add_argument(
    "--batch_size",
    "-b",
    type=int,
    default=128,
    help="input batch size for training (default: 128)",
)
group.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=10,
    help="number of epochs to train (default: 10)",
)
group.add_argument(
    "--label_smoothing",
    type=float,
    default=0.05,
    help="label smoothing value to use in loss function (default: 0.1)",
)
group.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    default=1e-3,
    help="learning rate value to use for optimizer (default: 1e-3)",
)
group.add_argument(
    "--use_mixed_precision",
    "-mp",
    type=bool,
    default=True,
    help="whether to use torch native mixed precision training (default: True)",
)


# Create Weights and Biases parameters
# TODO: make it so you can use W&B if you want to but don't have to if you dont want to... (this can come later, simple first)
group = parser.add_argument_group("Weights and Biases parameters")
group.add_argument(
    "--wandb-project",
    default="test_wandb_artifacts_by_reference",
    help="Weights & Biases project name",
)
group.add_argument(
    "--wandb-job-type",
    default="train and track food vision model",
    help="Weights & Biases job type",
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

# Misc parameters
group = parser.add_argument_group("Misc parameters")
group.add_argument(
    "--output",
    default="",
    type=str,
    help="path to output folder (default: none, current directory)",
)
group.add_argument(
    "--quick_experiment",
    "-qe",
    default=False,
    type=bool,
    help="whether to run a full run-through quick experiment (limits samples to first 100 units only) (default: False)",
)
group.add_argument(
    "--model_out_dir",
    default="models",
    type=str,
    help="path to model output directory (default: 'models')",
)

# Parse the args
def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__)

    return args, args_text


# TODO: perhaps the args could populate a config and then the config is used throughout the script?
args, args_text = _parse_args()

output_dir = Path(args.output)
# TODO: could log args to Wandb as a dict for the run?
with open(os.path.join(output_dir, "args.yaml"), "w") as f:
    f.write(args_text)

# Set devices
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Setup AMP (automatic mixed precision training)
from functools import partial

if args.use_mixed_precision:
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)
    loss_scaler = torch.cuda.amp.GradScaler()
    print("[INFO] Using mixed precision training. Training with dtype: torch.float16")
else:
    amp_autocast = suppress  # do nothing (no mixed precision)
    print(
        "[INFO] Not using mixed precision training. Training with dtype: torch.float32"
    )


# Set seeds
# TODO: could functionize this?
torch.manual_seed(42)
torch.cuda.manual_seed(42)


### Setup Artifacts ###
# TODO: should I load the Artifacts from another file? e.g. load_artifacts.py?
# TODO: can init Weights & Biases to log a config
# Configuration info: Log hyperparameters, a link to your dataset, or the name of the architecture you're using as config parameters, passed in like this: wandb.init(config=your_config_dictionary).
run = wandb.init(
    project=args.wandb_project, job_type=args.wandb_job_type, tags=["training"]
)

# Add args config to Weights & Biases
wandb.config.update(args)

dataset_at = run.use_artifact(args.wandb_dataset_artifact, type="dataset")
images_dir = (
    dataset_at.download()
)  # TODO: change this artifact to just be a standard GCP bucket of all images (rather than food_vision_199_classes_images:latest) -> all images in a single bucket (not 199_classes, just all images), then index with labels

print(f"[INFO] Images directory: {images_dir}")

### Get labels ###
import pandas as pd

# TODO: could add annotations to a function in a separate file
# Load in the annotations
labels_at = run.use_artifact(args.wandb_labels_artifact, type="labels")
labels_dir = labels_at.download()
labels_path = Path(labels_dir) / "annotations.csv"
print(f"[INFO] Labels path: {labels_path}")
annotations = pd.read_csv(labels_path)

# Create a dictionary of class_names and labels
class_names = annotations["class_name"].to_list()
class_labels = annotations["label"].to_list()
class_dict = dict(sorted(dict(zip(class_labels, class_names)).items()))
print(f"[INFO] Working with: {len(class_dict)} classes")
# Save class_dict to txt with each class on a new line
with open(os.path.join(output_dir, "class_dict.txt"), "w") as f:
    f.write(json.dumps(class_dict))

### Create dataset ###
# TODO: turn this into it's own script for loading the data
# Make a custom dataset reader for Timm to read images from a directory
from timm.data import ImageDataset
from timm.data.readers.reader import (
    Reader,
)


class FoodVisionReader(Reader):
    def __init__(
        self,
        image_root,
        label_root,
        class_to_idx,
        split="train",
        quick_experiment=args.quick_experiment,
    ):
        super().__init__()
        self.image_root = Path(image_root)

        # Get a mapping of classes to indexes
        self.class_to_idx = class_to_idx

        # Get a list of the samples to be used
        # TODO: could create the class_to_idx here? after loading the labels?
        # TODO: this would save opening the labels with pandas more than once...
        self.label_root = pd.read_csv(label_root)

        # Filter samples into "train" and "test"
        # TODO: add an index so I can select X amount of samples to use (e.g. for quick exerpimentation)
        # TODO: e.g. if args.quick_experiment == True: self.samples = self.samples[:100]
        if split == "train":
            self.samples = self.label_root[self.label_root["split"] == "train"][
                "image_name"
            ].to_list()
        elif split == "test":
            self.samples = self.label_root[self.label_root["split"] == "test"][
                "image_name"
            ].to_list()

        # Perform a quick training experiment on a small subset of the data
        if quick_experiment:
            self.samples = self.samples[:100]

    def __get_label(self, sample_name):
        return self.label_root.loc[self.label_root["image_name"] == sample_name][
            "label"
        ].values[0]

    def __getitem__(self, index):
        sample_name = self.samples[index]
        path = self.image_root / sample_name
        target = self.__get_label(sample_name)
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = Path(self.samples[index])
        if basename:
            filename = filename.parts[-1]
        elif not absolute:
            filename = self.image_root / filename
        return filename


# TODO: make the following arguments better in terms of how they're used
food_vision_reader = FoodVisionReader(
    image_root=images_dir,
    label_root=labels_path,
    class_to_idx=class_dict,
    split="train",
)

# Create tranforms
from timm.data import create_transform

# Create datasets
# TODO: maybe a good idea to print out how many samples are in each dataset?
train_dataset = ImageDataset(
    root=str(images_dir),
    reader=FoodVisionReader(
        images_dir, labels_path, class_to_idx=class_dict, split="train"
    ),
    transform=create_transform(input_size=args.image_size, is_training=True),
)

test_dataset = ImageDataset(
    root=str(images_dir),
    reader=FoodVisionReader(
        images_dir, labels_path, class_to_idx=class_dict, split="test"
    ),
    transform=create_transform(input_size=args.image_size, is_training=False),
)

# Create DataLoaders
from torch.utils.data import DataLoader

BATCH_SIZE = args.batch_size
NUM_WORKERS = args.workers

# TODO: make a sampler that samples X random samples from the dataset rather than the whole thing (for faster experimentation)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

### Create model ###
# TODO: fix the number of classes? (should come from args/annotations)
# TODO: add the modal name to a config/argparse?
# TODO: setup model config in W&B - https://docs.wandb.ai/guides/track/config
def create_model(model_name=args.model, num_classes=len(class_dict)):
    model = timm.create_model(
        model_name=model_name, pretrained=True, num_classes=num_classes
    )

    # Set all parameters to not requiring gradients
    for param in model.parameters():
        param.requires_grad = False

    # Set the last layer to require gradients (fine-tune the last layer only)
    for param in model.head.fc.parameters():
        param.requires_grad = True

    return model


model = create_model()
model.to(device)

# TODO: fix this setup and have optimizers in the config/argparse
from torch import nn

# TODO: setup mixed precision in train_step()
# TODO: fix engine script to work right within this script
from nutrify.engine import train, train_step, test_step
from tqdm.auto import tqdm

loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# TODO: make a "validate every... epoch" for example could validate every 5 epochs or something

from nutrify import utils


def train_one_epoch(
    epoch,
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    device=torch.device("cuda"),
    amp_autocast=suppress,
    loss_scaler=None,
):

    losses_meter = utils.AverageMeter()
    top1_meter = utils.AverageMeter()
    top5_meter = utils.AverageMeter()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    model.train()

    num_batches_per_epoch = len(train_dataloader)
    last_batch_idx = num_batches_per_epoch - 1

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
        last_batch = batch_idx == last_batch_idx  # Check to see if it's the last batch
        inputs, targets = inputs.to(device), targets.to(device)

        with amp_autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # TODO: update loss meter

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Calcuate train loss and train accuracy
        outputs_pred_label = torch.argmax(outputs, dim=1)
        train_loss += loss.item()
        train_acc += (outputs_pred_label == targets).sum().item() / len(targets)

        # Print out metrics
        if batch_idx == 0:
            # Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"batch: {batch_idx+1}/{num_batches_per_epoch} |"
                f"train_loss: {train_loss/len(train_dataloader):.4f} | "
                f"train_acc: {train_acc/len(train_dataloader):.4f} | "
            )

        elif last_batch or last_batch_idx % args.log_interval == 0:
            # Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"batch: {batch_idx+1}/{num_batches_per_epoch} |"
                f"train_loss: {train_loss/len(train_dataloader):.4f} | "
                f"train_acc: {train_acc/len(train_dataloader):.4f} | "
            )


# TODO: break this down into train_one_epoch (like timm's train.py)
# Train the model
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
      For example if training for epochs=2:
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
        )
        test_loss, test_acc = test_step(
            epoch=epoch,
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        # Print out what's happening
        print(
            f"Results Epoch {epoch+1}: "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # TODO: Log results to Weights & Biases (requires a dict!)
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

    # Return the filled results at the end of the epochs
    return results


# TODO: update this by potentially using the train_one_epoch function (create this first)
vanilla_pytorch_results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=args.epochs,
    device=device,
)

# Create a function to save to Google Storage
# TODO: make an export function to save the model to different store types
# TODO: put this file into a utils dir
from google.cloud import storage


def upload_to_gs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # # The ID of your GCS bucket
    # bucket_name = GS_BUCKET
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    print(f"[INFO] Google Cloud Bucket name: {bucket_name}")
    print(f"[INFO] Uploading {source_file_name} to {destination_blob_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"[INFO] Connected to bucket: {bucket_name}")
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"[INFO] File {source_file_name} uploaded to {destination_blob_name}.")
    print(f"[INFO] File size: {blob.size} bytes")

    # TODO: Make the blob public -- do I want this to happen?
    # blob.make_public()
    # print(f"[INFO] Blob public URL: {blob.public_url}")

    # print(f"[INFO] Blob download URL: {blob._get_download_url()}")

    return destination_blob_name


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.
    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    return model_save_path


# TODO: Functionize the model saving and upload model to GCS/W&B Artifacts?
# TODO: if we save the model to W&B Artifacts, we get the input/output Artifacts from a run (e.g. what data went in and what model came out)
### Save the model
model_export_dir = Path(args.model_out_dir)
# TODO: Save the model with some kind of time/name stamp
# Get the current time
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Save the model
# TODO: save this to GCP/Weights & Biases as an Artifact
model_export_name = f"{current_time}_model_{args.model}.pth"
model_save_path = save_model(
    model=model, target_dir=model_export_dir, model_name=model_export_name
)
# Upload the model to Google Storage
# TODO: Could I set this to a conditional to only upload if the model is better than the previous one?
# TODO: Results could be gathered from Weights & Biases on which model is the best...
# if args.upload_to_gs:
model_gcs_path = upload_to_gs(
    bucket_name=GS_BUCKET,
    source_file_name=model_save_path,
    destination_blob_name=str(model_save_path),
)

# TODO: Add the reference model log file from GCP to W&B Artifacts
# TODO: Could add this W&B model registry as well?
def log_model_artifact_to_wandb(model_path, project=args.wandb_project, run=run):
    """Logs a model to Weights & Biases as an artifact."""
    print(f"[INFO] Logging model to Weights & Biases...")
    # Create a wandb artifact
    model_artifact = wandb.Artifact(
        name="trained_model",
        type="model",
        description="A model trained on the Food Vision dataset.",
    )

    # Add the model file to the artifact
    model_artifact.add_file(model_path)

    # Save the model to W&B
    wandb.save(str(model_path))

    # TODO: Add reference file to GCP
    # model_artifact.add_reference(model_gcs_path)

    # Log the artifact to W&B
    # run = wandb.init(project=project)
    run.log_artifact(model_artifact)


# TODO: make this cleaner (e.g. the "run" parameter could be clearer, right now it's only set from the top)
# TODO: save the model's size (e.g. 54MB to W&B config to keep track of how big the model is)
# log_model_artifact_to_wandb(model_gcs_path=model_gcs_path, run=run)
log_model_artifact_to_wandb(model_path=model_save_path, run=run)


# Finish the W&B run
run.finish()
