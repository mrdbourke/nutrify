"""
Utility functions to perform zero-shot image classification with various models.

Many are based off the open_clip library: https://github.com/mlfoundations/open_clip
"""
import torch
import open_clip

from PIL import Image

from tqdm.auto import tqdm

from utils.misc import sort_dict_by_values

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

        image = open_clip_preprocess(Image.open(image_path)).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = open_clip_model.encode_image(image.to(device))
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