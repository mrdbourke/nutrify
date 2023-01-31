"""
Script for fixing FoodVision labels with Label Studio.

Takes a series of images and their predictions and sets up a Label Studio
instance to relabel the "most wrong" predictions.

TODO: Currently requires a Label Studio instance to be running, for example,
using "label-studio" in the command line. This should be automated in the
future.
"""
import argparse
import json
import os
import wandb
import yaml

import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

from utils.wandb_utils import wandb_load_artifact

# Create config parser (to get baseline config parameters)
config_parser = parser = argparse.ArgumentParser(description="Fix labels config file")
parser.add_argument("-c", "--config", default="configs.default_config", type=str, help="config file path (default: configs.default_config)")

# Create argument parser
parser = argparse.ArgumentParser(description="Fix FoodVision labels with Label Studio.")

# Create Weights & Biases arguments
group = parser.add_argument_group("Weights and Biases parameters")
group.add_argument(
    "--wandb-project",
    default="test_wandb_artifacts_by_reference",
    help="Weights & Biases project name",
)
group.add_argument(
    "--wandb-job-type",
    default="import_predictions_and_update_labels",
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
    default="creating Label Studio instance to fix 'most wrong' samples",
    help="Weights & Biases run notes, similar to writing a git commit message, 'what did you do?' (default '')",
)
group.add_argument(
    "--wandb-run-tags",
    default=["fix_labels_with_label_studio"],
    type=list,
    help="Weights & Biases run tags, for example (['predict_and_evaluate'])",
)
group.add_argument(
    "--wandb-pred-dataset-split",
    "-wb_split",
    default="test_predictions:latest",
    type=str,
    help="Weights & Biases dataset split, for example 'test_predictions:latest' or 'train_predictions:latest'",
)

# Create Misc arguments
group = parser.add_argument_group("Misc")
group.add_argument(
    "--n-most-wrong",
    "-n",
    default=20,
    type=int,
    help="Number of most wrong samples to find (default 20)",
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

# Parse arguments
args, args_text = _parse_args()

GS_BUCKET = args.gs_bucket_name
GS_IMAGE_STORAGE_PATH = args.gs_image_storage_path
PATH_TO_LABEL_STUDIO_API_KEY = args.path_to_label_studio_api_key

# TODO: could reproduce this for Google Storage Key + add the functions to a checks.py file or something?
# Check that the PATH_TO_LABEL_STUDIO_API_KEY is a JSON file that exists
def check_label_studio_api_key(path_to_label_studio_api_key):
    if not os.path.exists(path_to_label_studio_api_key):
        raise ValueError(
            f"PATH_TO_LABEL_STUDIO_API_KEY file not found, you passed {path_to_label_studio_api_key}."
        )
    # Check the file is a JSON file
    if not path_to_label_studio_api_key.endswith(".json"):
        raise ValueError(
            f"PATH_TO_LABEL_STUDIO_API_KEY file must be a JSON file, you passed {path_to_label_studio_api_key}."
        )

check_label_studio_api_key(args.path_to_label_studio_api_key)


def main():
    # Test GCP connection before proceeding
    # Connect to GCP
    from utils import set_gcp_credentials, test_gcp_connection
    set_gcp_credentials(path_to_key=args.path_to_gcp_credentials)
    test_gcp_connection()

    # Setup and check Label Studio connection
    label_studio_instance = check_connection_to_label_studio()

    # TODO: potentially put these in a config? e.g. config.py, see: https://github.com/mrdbourke/nutrify/issues/49
    WANDB_PROJECT = args.wandb_project
    WANDB_JOB_TYPE = args.wandb_job_type
    WANDB_DATASET = args.wandb_dataset_artifact
    WANDB_LABELS = args.wandb_labels_artifact
    WANDB_NOTES = args.wandb_run_notes
    WANDB_RUN_TAGS = args.wandb_run_tags
    WANDB_PREDICTIONS = args.wandb_pred_dataset_split
    N_MOST_WRONG = args.n_most_wrong

    # Add check for WANDB_PREDICTIONS
    if WANDB_PREDICTIONS not in ["test_predictions:latest", "train_predictions:latest"]:
        raise ValueError(
            f"WANDB_PREDICTIONS must be 'test_predictions:latest' or 'train_predictions:latest', you passed {WANDB_PREDICTIONS}."
        )

    # Print an info string of what's happening
    print(
        f"[INFO] Finding {N_MOST_WRONG} most wrong samples from '{WANDB_PREDICTIONS}' predictions."
    )

    # Setup Weights & Biases run
    run = wandb.init(
        project=WANDB_PROJECT,
        job_type=WANDB_JOB_TYPE,
        tags=WANDB_RUN_TAGS,
        notes=WANDB_NOTES,
    )

    # Update Weights & Biases run with arguments
    wandb.config.update(args)

    # Get images
    images_dir = wandb_load_artifact(
        wandb_run=run, artifact_name=WANDB_DATASET, artifact_type="dataset"
    )
    print(f"[INFO] Images directory: {images_dir}.")

    # Download labels artifact
    from utils.wandb_utils import wandb_download_and_load_labels

    annotations, class_names, class_dict, reverse_class_dict, labels_path = wandb_download_and_load_labels(wandb_run=run, 
                                                                                                            wandb_labels_artifact_name=args.wandb_labels_artifact)

    # Download the predictions DataFrame
    df = download_and_load_predictions_from_wandb(
        wandb_run=run, artifact_name=WANDB_PREDICTIONS, artifact_type="predictions"
    )

    # Find the most wrong
    most_wrong = find_n_most_wrong(df, n_most_wrong=N_MOST_WRONG)

    # Convert most wrong DataFrame to Label Studio JSON format
    most_wrong_dict = most_wrong.to_dict("records")

    # TODO: Turn this into a function to export DataFrames to Label Studio JSON format
    label_studio_most_wrong = []
    for sample in tqdm(most_wrong_dict):
        storage_path = GS_IMAGE_STORAGE_PATH

        # Get image parameters
        image_name = Path(sample["image_path"]).name
        image_path = str(Path(storage_path, image_name))
        prediction_class = sample["pred_class"]
        true_class = sample["true_class"]
        prediction_score = sample["pred_prob"]
        top_n_classes = sample["top_n_preds"]
        data_split = sample["split"]

        # Turn top_n_classes into a list of dictionaries from a string (make sure the JSON can parse single quotes)
        top_n_classes = top_n_classes.replace("'", '"')
        top_n_classes = json.loads(top_n_classes)

        # Convert top_n_classes to the following format: {pred_class: "pred_class", pred_prob: "pred_prob"} (where pred_prob has 4 decimal places)
        for item in top_n_classes:
            item["class"] = item.pop("pred_class")
            item["prob"] = round(item.pop("pred_prob"), 4)
            del item["pred_label"]

        # Turn top_n_classes into a string of tuples with just the class and probability
        top_n_classes = [f"{item['class']}: {item['prob']}" for item in top_n_classes]

        # Create the Label Studio JSON sample
        label_studio_sample = {
            "data": {
                "image": image_path,
                "true_string": f"true: {true_class}",
                "pred_string": f"pred: {prediction_class}",
                "score_string": f"score: {prediction_score:.3f}",
                "top_n_classes_string": f"top_n_classes (class: pred_prob): {top_n_classes}",
                "data_split_string": f"data_split: {data_split}",
            },
            "predictions": [
                {
                    "model_version": "TODO: add model version",
                    "score": prediction_score,
                    "result": [
                        {
                            "type": "choices",
                            "from_name": "label",  # This is the name for "Choices" tag in the Label Studio UI, e.g. <Choices name="label".../>
                            "to_name": "image",
                            "value": {"choices": [prediction_class]},
                        }
                    ],
                }
            ],
        }
        label_studio_most_wrong.append(label_studio_sample)

    ## Create Label Studio Interface
    ## TODO: optionally save the Label Studio JSON to a file
    # save code here...

    # Turn class names into Label Studio format
    template_string = '<Choice value="{}"/>'
    class_names_html = []
    for class_name in class_names:
        class_names_html.append(template_string.format(class_name))

    # Turn class_names_html into a single string with newlines separating each class
    class_names_html_string = "\n".join(class_names_html)

    # TODO: optionally save class_names_html_string to file?
    # save code here...

    # Create Label Studio HTML interface
    # (TODO: could put this in a file?), such as a config file, e.g. foodvision/classification/label_studio_interface.py
    start_html_string = """<View>
    <Image name="image" value="$image" maxWidth="750px"/>
    <Text name="true_string" value="$true_string"/>
    <Text name="pred_string" value="$pred_string"/>
    <Text name="score_string" value="$score_string"/>
    <Text name="top_n_classes_string" value="$top_n_classes_string"/>
    <Text name="data_split_string" value="$data_split_string"/>

    <Header value="Is the image clear or confusing?" />
    <Text name="clear or confusing instruction" value="A clear image would be a photo where the food is clearly visible, where as a confusing image may have a lot going on."/>
    <Choices name="clear_or_confusing" toName="image" showInline="true">
        <Choice value="clear"/>
        <Choice value="confusing"/>
    </Choices>

    <Header value="Is the image a whole food or a dish?" />
    <Text name="whole food or dish" value="A 'whole food' would be a single food/something you wouldn't need to cook (e.g. an apple or a bowl of berries), where as a dish would be something you'd prepare (e.g. a bowl of ramen or sandwich)."/>
    <Choices name="whole_food_or_dish" toName="image" showInline="true">
        <Choice value="whole_food"/>
        <Choice value="dish"/>
    </Choices>

    <Header value="Is there one food item or multiple?" />
    <Text name="one food or multuple description" value="One food would be a photo of an apple and only an apple or a banana and only a banana (one food also counts when there is multiple bananas but only bananas), where as, multiple would be a photo of blueberries, strawberries and blackberries (or a dish)."/>
    <Choices name="one_food_or_multiple" toName="image" showInline="true">
        <Choice value="one_food"/>
        <Choice value="multiple_foods"/>
    </Choices>

    <Header value="Which class best suits the image?" />
    <Text name="best suited class" value="In the case where multiple classes could suit the image, choose the one which is most dominant (the first one which comes to mind)."/>
    <Choices name="label" toName="image" showInline="true">
    """

    # # Open class_names_html.txt as a single string
    # with open("class_names_html.txt", "r") as f:
    #     choices_string = f.read()

    end_html_string = """
    </Choices>
    </View>
    """

    # Add the strings together
    label_studio_labeling_interface_html_string = (
        start_html_string + class_names_html_string + end_html_string
    )

    # TODO: optionally save the string to text file
    # # Save the string to a text file
    # with open("label_studio_labeling_interface_test.html", "w") as f:
    #     f.write(label_studio_labeling_interface_html_string)

    return (
        run,
        label_studio_instance,
        label_studio_most_wrong,
        label_studio_labeling_interface_html_string,
    )

def download_and_load_predictions_from_wandb(
    wandb_run, artifact_name, artifact_type="predictions"
):  
    from utils.wandb_utils import wandb_load_artifact
    predictions_dir = wandb_load_artifact(
        wandb_run=wandb_run, artifact_name=artifact_name, artifact_type=artifact_type
    )

    print(f"[INFO] Predictions dir: {predictions_dir}")

    # TODO: make this possible for train/test sets (could merge the CSV files?)
    csv_files = list(Path(predictions_dir).glob("*.csv"))[0]
    print(f"[INFO] CSV files: {csv_files}")

    # Load the csv from predictions_dir
    df = pd.read_csv(csv_files)

    return df


# Find the top 100 examples in a DataFrame where the model was most confident but wrong
def find_n_most_wrong(
    df,
    n_most_wrong=100,
    true_label="true_label",
    pred_label="pred_label",
    pred_prob="pred_prob",
):
    # Find the top n_most_wrong examples where the model was most confident but wrong
    df_wrong = df[df[pred_label] != df[true_label]]
    return df_wrong.sort_values(pred_prob, ascending=False).head(n_most_wrong)


# TODO: add parameters to this
def check_connection_to_label_studio():
    ## TODO: Send the Label Studio JSON tasks and HTML Interface to Label Studio
    from label_studio_sdk import Client

    ## TODO: make this available to run locally or on some host?
    # see here for hosting: https://labelstud.io/guide/start.html#Run-Label-Studio-with-an-external-domain-name
    LABEL_STUDIO_URL = "http://localhost:8080"  # TODO: this should be an arg?

    # This loads from .env file in VS Code, see here: https://code.visualstudio.com/docs/python/environments#_environment-variables
    # TODO: fix this loading from env environment:
    # LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    # Get Label Studio API key from file
    with open(PATH_TO_LABEL_STUDIO_API_KEY) as f:
        data = json.load(f)
        LABEL_STUDIO_API_KEY = data["LABEL_STUDIO_API_KEY"]

    # # TODO: could potentially run Label Studio as a subprocess (in the background)
    # import subprocess

    # # Start Label Studio
    # subprocess.run(["label-studio"])

    # Connect to Label Studio (if possible)
    try:
        label_studio_instance = Client(
            url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY
        )
        label_studio_instance.check_connection()
        print("[INFO] Label Studio is running. Data can be uploaded/modified.")
    except Exception as e:
        print(
            "Error: Label Studio is not running. Please start Label Studio `label-studio` and try again."
        )
        raise e

    return label_studio_instance


def export_tasks_to_label_studio(label_studio_instance, label_config, label_tasks):
    import datetime

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    # TODO: add info on where the Label Studio interface is available and how to access it
    project = label_studio_instance.start_project(
        title=f"{now}_food_vision_fix_labels",
        label_config=label_config,
    )

    # TODO: print out statistics about the most wrong labels
    print(f"[INFO] Number of tasks being sent to Label Studio: {len(label_tasks)}.")
    project.import_tasks(tasks=label_tasks)

    # TODO: could I connect Google Storage as an export option here?
    # see here: https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.connect_google_export_storage
    # Note: this requires having Google Application Credentials setup
    # TODO: add print out of where the bucket is going to (e.g. bucket name and prefix)
    project.connect_google_export_storage(
        bucket=GS_BUCKET,
        prefix=f"label_studio_exports/classification",
        google_application_credentials=args.path_to_gcp_credentials,
    )

    # Print out the URL of Label Studio to access the access the tasks
    print(f"[INFO] View and work on Label Studio project at {project.url}")


if __name__ == "__main__":
    (
        run,
        label_studio_instance,
        label_studio_most_wrong,
        label_studio_labeling_interface_html_string,
    ) = main()

    # Export the tasks to Label Studio
    export_tasks_to_label_studio(
        label_studio_instance=label_studio_instance,
        label_config=label_studio_labeling_interface_html_string,
        label_tasks=label_studio_most_wrong,
    )

    # Finish the Weights & Biases run once main is done
    run.finish()
