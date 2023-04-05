"""
File to create various models for FoodVision, Food Not Food etc.
"""

import torch
import timm

from torch import nn

def create_coatnext_nano_rw_224_model(pretrained: bool, 
                                      num_classes: int,
                                      train_body: bool = False) -> nn.Module:
    """
    Create a model from the timm library.
    """
    model = timm.create_model(model_name="coatnext_nano_rw_224", 
                              pretrained=pretrained,
                              num_classes=num_classes)
    
    if train_body:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
    
    # Try an extra layer on top 
    in_features = model.head.fc.in_features
    model.head.fc = nn.Sequential(
        nn.Linear(in_features=in_features, 
                  out_features=in_features),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=in_features,
                  out_features=num_classes)
    )

    # Set the last layer to require gradients (fine-tune the last layer only)
    for param in model.head.fc.parameters():
        param.requires_grad = True

    # Hard code the image size
    image_size = 224 
    
    return model, image_size


def create_eva02_small_patch14_336_model(pretrained: bool, 
                                         num_classes: int,
                                         train_body: bool = False) -> nn.Module:
    """
    Create EVA-02 small patch 14 336 model from timm library.

    See link: https://github.com/baaivision/EVA/tree/master/EVA-02 
    """

    # Requires timm v0.8.17+
    model_name = "eva02_small_patch14_336.mim_in22k_ft_in1k"

    model = timm.create_model( 
        model_name=model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
    )

    if train_body:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

    # Set the last layer to require gradients (fine-tune the last layer only)
    for param in model.head.parameters():
        param.requires_grad = True

    # Hard code the image size
    image_size = 336 

    return model, image_size

# Create a dictionary of models to use
model_dict = {
    "coatnext_nano_rw_224": create_coatnext_nano_rw_224_model,
    "eva02_small_patch14_336": create_eva02_small_patch14_336_model
}