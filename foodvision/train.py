"""
Training script for FoodVision models.
TODO: adapt the script to use more elegant features such as in: 
* Timm training script: https://github.com/rwightman/pytorch-image-models/blob/main/train.py 
* Timm training script guide: 
* Potentially could put the model into a Hugging Face Transformer and then use the model
    with the Transformers class?

Style guide: https://google.github.io/styleguide/pyguide.html 
"""
import argparse
import json
import os
from contextlib import \
    suppress  # suppresses amp_autocast if it's not available
from pathlib import Path
from typing import Dict, List
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader

import timm
from timm.data import ImageDataset, create_transform

import wandb
from data_loader import FoodVisionReader

# Connect to GCP
from utils.gcp_utils import set_gcp_credentials, test_gcp_connection
set_gcp_credentials(path_to_key="utils/google-storage-key.json")
test_gcp_connection()

# Create config parser
config_parser = parser = argparse.ArgumentParser(description="Training config file")
parser.add_argument("-c", "--config", default="configs.default_config", type=str, help="config file path (default: configs.default_config)")

# Create regular parser
parser = argparse.ArgumentParser(description="Train a FoodVision model.")

# Create dataset parameters
group = parser.add_argument_group("Dataset parameters")
group.add_argument("--polars_or_pandas", "-p", default="pandas", type=str, help="use pandas or polars to load DataFrame for DataLoader")
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
group.add_argument(
    "--auto_augment",
    "-aa",
    default=True,
    type=bool,
    help="whether to use auto augmentation, default value is 'rand-m9-mstd0.5', see https://timm.fast.ai/RandAugment for more (default: True)",
)
group.add_argument(
    "--pin_memory",
    default=True,
    type=bool,
    help="whether to pin memory for dataloader (default: True)",
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
# TODO: add num_classes? as arg - this could be inferred from data?

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
    default=0.1,
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
group.add_argument(
    "--wandb-run-notes",
    "-wb_notes",
    default="",
    help="Weights & Biases run notes, similar to writing a git commit message, 'what did you do?' (default '')",
)

# Misc parameters
group = parser.add_argument_group("Misc parameters")
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
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

# TODO: does this have to be here?
output_dir = Path(args.output)
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

if args.auto_augment:
    print(f"[INFO] Using auto augment, strategy: {args.auto_augment}")
    train_transform = create_transform(input_size=args.image_size, is_training=True, auto_augment="rand-m9-mstd0.5")
else:
    print("[INFO] Not using auto augment, normal image transformations will be used.")
    train_transform = create_transform(input_size=args.image_size, is_training=True)

# Set seeds
from utils import seed_everything
seed_everything(args.seed)


### Setup Artifacts ###
# TODO: should I load the Artifacts from another file? e.g. load_artifacts.py?
# TODO: can init Weights & Biases to log a config
# Configuration info: Log hyperparameters, a link to your dataset, or the name of the architecture you're using as config parameters, passed in like this: wandb.init(config=your_config_dictionary).
run = wandb.init(
    project=args.wandb_project,
    job_type=args.wandb_job_type,
    tags=["training"], # TODO set this on args
    notes=args.wandb_run_notes,
)

# Add args config to Weights & Biases
wandb.config.update(args)

from utils.wandb_utils import wandb_load_artifact, wandb_download_and_load_labels

run = wandb.init(project=args.wandb_project, 
                 job_type=args.wandb_job_type,
                 tags=args.wandb_run_tags,
                 notes=args.wandb_run_notes)

images_dir = wandb_load_artifact(
    wandb_run=run, 
    artifact_name=args.wandb_dataset_artifact, 
    artifact_type="dataset")

print(f"[INFO] Images directory: {images_dir}")

annotations, class_names, class_dict, reverse_class_dict, labels_path = wandb_download_and_load_labels(wandb_run=run,
wandb_labels_artifact_name=args.wandb_labels_artifact)

# Create datasets
# TODO: maybe a good idea to print out how many samples are in each dataset?
polars_or_pandas = args.polars_or_pandas
if polars_or_pandas == "polars":
    from foodvision.data_loader import FoodVisionReaderPolars
    print("[INFO] Using Polars for DataFrame Loading")
    FoodVisionReader = FoodVisionReaderPolars
else:
    print("[INFO] Using Pandas for DataFrame Loading")

train_dataset = ImageDataset(
    root=str(images_dir),
    parser=FoodVisionReader(
        images_dir, labels_path, class_to_idx=class_dict, split="train",
        quick_experiment=args.quick_experiment
    ),
    transform=train_transform,
)

test_dataset = ImageDataset(
    root=str(images_dir),
    parser=FoodVisionReader(
        images_dir, labels_path, class_to_idx=class_dict, split="test",
        quick_experiment=args.quick_experiment
    ),
    transform=create_transform(input_size=args.image_size, is_training=False),
)

# Create DataLoaders
from torch.utils.data import DataLoader

BATCH_SIZE = args.batch_size
NUM_WORKERS = args.workers
PIN_MEMORY = args.pin_memory

print(f"[INFO] Using batch size: {BATCH_SIZE}")
print(f"[INFO] Using num workers: {NUM_WORKERS}")
print(f"[INFO] Using pin memory: {PIN_MEMORY}")

# TODO: make a sampler that samples X random samples from the dataset rather than the whole thing (for faster experimentation)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

### Create model ###
# TODO: fix the number of classes? (should come from args/annotations)
# TODO: add the modal name to a config/argparse?
# TODO: setup model config in W&B - https://docs.wandb.ai/guides/track/config
def create_model(model_name=args.model, 
                 pretrained=args.pretrained,
                 num_classes=len(class_dict)):

    model = timm.create_model( 
        model_name=model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
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
from tqdm.auto import tqdm

# TODO: setup mixed precision in train_step()
# TODO: fix engine script to work right within this script
from engine import test_step, train, train_step

loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# TODO: make a "validate every... epoch" for example could validate every 5 epochs or something

# TODO: make this function usable (or similar to timm's train.py)
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
    amp_autocast: torch.cuda.amp.autocast = suppress,
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
print("[INFO] Training model...")
vanilla_pytorch_results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=args.epochs,
    device=device,
    amp_autocast=amp_autocast,
)

# Create a function to save to Google Storage
# TODO: make an export function to save the model to different store types
from utils.gcp_utils import upload_to_gs

# TODO: move this into a model utils?
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
from utils import get_now_time

# Save the model
# TODO: save this to GCP/Weights & Biases as an Artifact
current_time = get_now_time()
model_export_name = f"{current_time}_model_{args.model}.pth"
model_save_path = save_model(
    model=model, target_dir=model_export_dir, model_name=model_export_name
)
# Upload the model to Google Storage
# TODO: Could I set this to a conditional to only upload if the model is better than the previous one?
# TODO: Results could be gathered from Weights & Biases on which model is the best...
# if args.upload_to_gs:
model_gcs_path = upload_to_gs(
    bucket_name=args.gs_bucket_name,
    source_file_name=model_save_path,
    destination_blob_name=str(model_save_path),
)

# TODO: Add the reference model log file from GCP to W&B Artifacts
# TODO: Could add this W&B model registry as well?
# TODO: move this wandb utils 
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

# Get model_save_path file size in MB
model_save_path_size_in_mb = model_save_path.stat().st_size / 1e6

# Log the model_save_path_size_in_mb to W&B
run.log({"model_save_path_size_in_mb": model_save_path_size_in_mb})


# TODO: make this cleaner (e.g. the "run" parameter could be clearer, right now it's only set from the top)
# TODO: save the model's size (e.g. 54MB to W&B config to keep track of how big the model is)
# log_model_artifact_to_wandb(model_gcs_path=model_gcs_path, run=run)
log_model_artifact_to_wandb(model_path=model_save_path, run=run)


# Finish the W&B run
run.finish()