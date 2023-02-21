"""
Utility functions to perform zero-shot image classification with various models.

Many are based off the open_clip library: https://github.com/mlfoundations/open_clip
"""
from typing import List

import torch
import open_clip

from PIL import Image

from tqdm.auto import tqdm

from utils.misc import sort_dict_by_values, open_image

open_clip_model_name = "ViT-H-14"
open_clip_pretrained = "laion2b_s32b_b79k"

# open_clip_model_name = 'convnext_large_d' 
# open_clip_pretrained = 'laion2b_s26b_b102k_augreg'

# open_clip_model_name = 'ViT-g-14'
# open_clip_pretrained = 'laion2b_s12b_b42k'

def create_open_clip_model_and_preprocess(model_name: str, pretrained: str, device: torch.device):
    open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    open_clip_model.to(device)

    # Get the tokenizer
    open_clip_tokenizer = open_clip.get_tokenizer(model_name)

    return open_clip_model, open_clip_preprocess, open_clip_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
open_clip_model, open_clip_preprocess, open_clip_tokenizer = create_open_clip_model_and_preprocess(open_clip_model_name, open_clip_pretrained, device)

def open_clip_compute_image_features_of_list(image_paths: list,
                                             open_clip_model=open_clip_model,
                                             open_clip_preprocess=open_clip_preprocess,
                                             device=device):
    image_features_list = []                                    
    for image_path in tqdm(image_paths, desc="Computing OpenCLIP image features"):
        image_feature_dict = {}
        image = open_image(image_path)
        image_processed = open_clip_preprocess(image).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = open_clip_model.encode_image(image_processed.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)       

        # Append image_path to dict
        image_feature_dict["image_path"] = image_path
        image_feature_dict["image_features"] = image_features
        image_features_list.append(image_feature_dict)
    
    return image_features_list

def open_clip_compute_text_features(text: list,
                                    open_clip_model=open_clip_model,
                                    open_clip_tokenizer=open_clip_tokenizer,
                                    device=device):

    text = open_clip_tokenizer(text)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = open_clip_model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features

def open_clip_zero_shot_classification(image_features, 
                                       text_features, 
                                       class_names, 
                                       device=device, 
                                       sorted=True):
                                       
    # Send features to device
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        similarity_values = (image_features @ text_features.T)[0] / 0.01
        probabilities = torch.softmax(similarity_values, dim=-1).cpu().tolist() # return values to CPU

        # Create a dict of class names and their probabilities
        class_name_similarity_probability_dict = dict(zip(class_names, probabilities))

    if sorted:
        return sort_dict_by_values(class_name_similarity_probability_dict)
    else:
        return class_name_similarity_probability_dict

def open_clip_get_image_and_text_similarity_dicts(open_clip_image_features_list: list,
                                                  target_classes: List[str],
                                                  max_len_of_similarity_dict: int = 10):
    
    # Set CLIP and BLIP to use GPU for creating image features if available
    # img_feature_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode text features on CPU (this is quick even for 100s of classes)
    # blip_text_features = blip_get_text_features(text=target_classes, device="cpu") # only get dims for zero-shot classification
    open_clip_text_features = open_clip_compute_text_features(text=target_classes)
    
    # Loop through image paths
    auto_created_labels = []
    for image_feature_dict in tqdm(open_clip_image_features_list, desc="Calculating image features and similarity dicts"):

        image_path = image_feature_dict["image_path"]
        # Match image path to pre-computed image features
        # print(image_paths.index(image_path))
        open_clip_img_features = image_feature_dict["image_features"]
        # blip_img_features = blip_image_features_list[image_paths.index(image_path)]["image_features"][:, 0] # TODO: could the indexing be cleaner?
        
        # Get sorted similarity dicts
        open_clip_similarity_dict = open_clip_zero_shot_classification(open_clip_img_features, open_clip_text_features, target_classes, sorted=True)
        # blip_similarity_dict = blip_zero_shot_classification(blip_img_features, blip_text_features, target_classes, sorted=True)

        # # Average the values of the similarity dicts
        # avg_similarity_dict = {}
        # for key in open_clip_similarity_dict.keys():
        #     avg_similarity_dict[key] = (open_clip_similarity_dict[key] + blip_similarity_dict[key]) / 2
        
        # # Sort the average similarity dict and reduce length to 5
        # avg_similarity_dict = dict(list(sort_dict_by_values(avg_similarity_dict).items())[:5])

        # If the length of the similarity dicts is max_len_of_similarity_dict, shorten it
        if len(open_clip_similarity_dict) > max_len_of_similarity_dict:
            open_clip_similarity_dict = dict(list(open_clip_similarity_dict.items())[:max_len_of_similarity_dict])
        
        # if len(blip_similarity_dict) > 5:
        #     blip_similarity_dict = dict(list(blip_similarity_dict.items())[:5])
        
        # Append to list
        auto_created_labels.append({"image_path": image_path,
                                    # "blip_sorted_similarity_dict": blip_similarity_dict,
                                    # "blip_top_1_class_name": list(blip_similarity_dict.keys())[0],
                                    # "blip_top_1_similarity_score": list(blip_similarity_dict.values())[0],
                                    "open_clip_sorted_similarity_dict": open_clip_similarity_dict,
                                    "open_clip_top_1_class_name": list(open_clip_similarity_dict.keys())[0],
                                    "open_clip_top_1_similarity_score": list(open_clip_similarity_dict.values())[0],
                                    # "avg_sorted_similarity_dict": avg_similarity_dict,
                                    # "avg_top_1_class_name": list(avg_similarity_dict.keys())[0],
                                    # "avg_top_1_similarity_score": list(avg_similarity_dict.values())[0]
                                })
    
    return auto_created_labels