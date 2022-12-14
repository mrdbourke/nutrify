# Scripts and modules for training/evaluating models

Foodvision = the ML model that powers Nutrify's computer vision system. 

Going to store scripts in here for training models and evaluating them.

The goal will be to have a dataflywheel happening from the main `train.py` script.

For example:
* `python train.py` - trains model (going to start by basing this off `timm`'s train script - https://github.com/rwightman/pytorch-image-models/blob/main/train.py)
* `evaluate.py` - will evaluate the model across a given dataset (that way we can inspect the worst performing classes/data)
* `load_data.py` - or something similar perhaps `data_loading/...` will have the data loading scripts
* `predict.py` - can perform predictions on given samples