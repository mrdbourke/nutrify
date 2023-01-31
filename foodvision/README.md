# FoodVision: the computer vision system powering Nutrify

This folder contains all of the computer vision related code for training and evaluating FoodVision models.

🍔👁️ FoodVision = the name of the computer vision model powering Nutrify.

The main goal of this repo/directory is to create a data engine for food images:

<img src="https://user-images.githubusercontent.com/16750345/215626025-bde8bcf3-eccb-4695-a330-6532767f5bbe.png" alt="nutrify data engine flow chart" title="Nutrify data engine flow chart January 2023">

## Overview

As of Janaury 2023, the workflow is focused on *multi-class classification*.

* **Inputs:** an image.
* **Outputs:** a single class of what food is in that image (e.g. "banana" or "apple" only).

Future efforts will likely break this functionality down into:
* `classification/` - for image classification (currently here).
* `detection/` - for object detection (future).
* `segmentation/` - for segmentation (future).

## What does each script do?

**Note:** These are subject to change (and likely will).

One-liner description (sorted alphabetically):

* `configs/` - stores different model/training/data config values in `default_config.py` these are imported in various scripts such as `train.py`.
* `utils/` - contains various utility functions used throughout top-level scripts (including utilities for Google Cloud + Weights & Biases), also contains keys for Google Cloud and Label Studio (stored locally only).
* `autocorrect_and_merge_labels.py` - finds the most-wrong samples discovered by `evaluate.py` and runs zero-shot image classification models (currently BLIP + CLIP) over the samples to detect if their labels could be corrected then automatically updates the labels & tracks them (**note:** autocorrection only happens on *training* data not *testing* data).
* `data_loader.py` - loads data into a PyTorch Dataset ready to be used for `train.py` script.
* `engine.py` - contains training and evaluation loops for `train.py`, though this may be replaced in the future by a simpler setup in `train.py` only.
* `evalute.py` - takes a trained model (output of `train.py`) and evaluates it across train/test sets, outputs prediction results and stores them on Google Storage and Weights & Biases for further evaluation.
* `fix_labels.py` - takes the results of `evaluate.py` and sets up a Label Studio instance for **manually** correcting labels, outputs fixed labels to a Google Storage bucket.
* `merge_labels_from_label_studio.py` - takes outputs of `fix_labels.py` (Label Studio-style JSON labels) and formats them to be merged with the original labels (currently a CSV file, though maybe in the future it will be a database of some sort) then tracks the updated labels for future runs.
* `train.py` - trains a FoodVision model, results and config are tracked with Weights & Biases experiments, outputs a model file to Google Storage (also tracked in Weights & Biases).
* `train_and_eval.sh` - runs `train.py` and `evaluate.py` sequentially with default configs.

**Note:** The trend here is that as much information as possible is tracked in:
* Weights & Biases Experiments (for model training results).
* Weights & Biases Artifacts (for data/labels/model files).
* Weights & Biases Tables (for data visualization).
* Google Storage (for a single source of truth of *most* files).

## Current workflow

The scripts can generally be run in any order (some will automatically exit, such as, `fix_labels.py` if no labels are found to be fixed).

But the default order would be:
- `train.py` - train a model on the stored images/labels, track results + model in Weights & Biases.
- `evalute.py` - evaluate the trained model on the given data, track results in Weights & Biases.
- `fix_labels.py` - export "most wrong" (samples where the prediction probability is high but the model is wrong) labels to Label Studio to be manually fixed if needed (by a human) and then saved in Google Storage.
- `merge_labels_from_label_studio.py` - takes outputs of `fix_labels.py` and automatically merges the corrected labels with the originals in Google Storage (this will be tracked so subsequent runs of `train.py` will use the new labels.
   - **Note:** After running `merge_labels_from_label_studio.py`, if there are any updates to the label/data files, these will be tracked in Weights & Biases and `train.py` will by default use the latest versions (this can be changed in the configs/args).
- `autocorrect_and_merge_labels.py` - some samples may have labels that are easy enough to *not* have to be corrected by a human, this script will use vision-language models to similarity match the text of the `class_names` (such as "apple", "banana") to a target image and will update the original label if both (currently using CLIP & BLIP from [LAVIS](https://github.com/salesforce/LAVIS) for redunancy) top-1 outputs match.
   - **Note:** Similar to above, after running `autocorrect_and_merge_labels.py`, the original label file may change (this will get tracked in Weights & Biases), this means subsequent runs of `train.py` will use the latest and most up-to-date version of the labels by default (this can be changed in the configs/args).
   
 **Note:** By default all scripts use the `:latest` version of each dataset/labels/models by loading them from Weights & Biases Artifacts. This means if a change is made to any dataset/labels/models, it is always tracked/versioned and the most recent version loaded by default (though this doesn't mean it is the *best* version).

For now, this works for the very *specific* use case of Nutrify's multi-class classification.

However, the overall steps would be similar for a different type of problem (detection/segmentation), just with different data structures. 
   
## Example usage

Train a model with default config:

```
python train.py
```

Train a model for 20 epochs (see args for more training options):

```
python train.py --epochs=20
```

Evaluate a model:

```
python evaluate.py
```

Fix labels (requires an instance of `label-studio` running):

```
# Start label-studio
label-studio

# Fix labels manually
python fix_labels.py
```

Merge manually fixed labels into original labels (exits if no fixed labels exist):

```
python merge_labels_from_label_studio.py
```

Autocorrect labels with vision-language models and merge them back into original labels:

```
python autocorrect_and_merge_labels.py
```

## TODO

These are roughly in order of necessity.

- [ ] clean scripts to make docstrings + workflow better (e.g. use and reuse util functions), see: https://github.com/mrdbourke/nutrify/issues/61 
- [ ] create `export.py` to export models in a particular format (e.g. PyTorch -> CoreML), see: https://github.com/mrdbourke/nutrify/issues/62 
- [ ] make scripts simpler (e.g. merge `engine.py` + `train.py` so things are less all over the place, see: https://twitter.com/karpathy/status/1620103412686942208?s=20&t=QSOJ2H2jWIElEWpNW2bLXg), see: https://github.com/mrdbourke/nutrify/issues/63 
- [ ] data collection pipeline (what happens when more data comes in? e.g. automatically label + train + evaluate), see: https://github.com/mrdbourke/nutrify/issues/64 
- [ ] autolabelling pipeline (e.g. take raw images in, label them if they don't have it) 
- [ ] clean labels pipeline (use cleanlab to clean busted labels), see: https://github.com/mrdbourke/nutrify/issues/52 
- [ ] deduplication pipeline (if working with internet-scale images, good thing to remove duplicates to prevent data leakage), see: https://github.com/mrdbourke/nutrify/issues/53 
- [ ] image generation pipeline (generate food images with Stable Diffusion or other similar models to enhance training data), https://github.com/mrdbourke/nutrify/issues/65

## Inspiration

This folder is built on the shoulders of giants:

* `timm` (Torch Image Models) - https://github.com/rwightman/pytorch-image-models
* yolov8 by ultralytics - https://github.com/ultralytics/ultralytics 
* Config ideas - https://twitter.com/i/web/status/1531155343505489920 
