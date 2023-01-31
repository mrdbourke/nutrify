# Scripts and modules for training/evaluating models

<img src="https://user-images.githubusercontent.com/16750345/215626025-bde8bcf3-eccb-4695-a330-6532767f5bbe.png" alt="nutrify data engine flow chart" title="Nutrify data engine flow chart January 2023">

TODO: update workflow
* [] add image of flow chart
* [] add steps to readme
    * [] write about how the args work (e.g. `default_config.py` is the main source but can be overidden with args)
* [] go through each script and make sure the workflow/loading is updated to follow the rest (e.g. load utils/other functions)
* [] add docstrings to code + utils for better usage



---

Current workflow:
    - `train.py` → `[evaluate.py](http://evaluate.py)` → `fix_labels.py` → fix labels in Label Studio interface → Save to GCP (auto) → `04_update_and_merge_manual_corrected_labels.ipynb` pulls labels from GCP → merges labels to original annotations → deletes and cleans up (updated annotations from local and Google Storage) → `05_update_and_merge_autocorrected_labels.ipynb`
        - Manual = requires human in the loop, auto = can be done with pure modelling alone

---

Foodvision = the ML model that powers Nutrify's computer vision system. 

Going to store scripts in here for training models and evaluating them.

The goal will be to have a dataflywheel happening from the main `train.py` script.

For example:
* `python train.py` - trains model (going to start by basing this off `timm`'s train script - https://github.com/rwightman/pytorch-image-models/blob/main/train.py)
* `evaluate.py` - will evaluate the model across a given dataset (that way we can inspect the worst performing classes/data)
* `load_data.py` - or something similar perhaps `data_loading/...` will have the data loading scripts
* `predict.py` - can perform predictions on given samples