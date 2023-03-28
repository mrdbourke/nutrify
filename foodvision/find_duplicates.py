"""
File to find duplicates in a target directory and store them somewhere.
"""

import os 

from pathlib import Path
from tqdm.auto import tqdm

import fastdup

import faulthandler

faulthandler.enable()

# TODO: make this arg
# images_dir = "notebooks/2023-02-22-food_photos"
images_dir = "./artifacts/food_vision_199_classes_images:v23"

image_paths = list(Path(images_dir).rglob("*.jp*g"))
print(f"[INFO] Finding duplicate images in: {images_dir}")
print(f"[INFO] Number of images: {len(image_paths)}")

# # Delete image cd6bcc10-0084-41bf-9063-8606453f222f.jpeg from images_dir
# # This image is causing an error in fastdup
# for image in os.listdir(images_dir):
#     if image == "cd6bcc10-0084-41bf-9063-8606453f222f.jpeg":
#         os.remove(os.path.join(images_dir, image))

# # Remove image cd6bcc10-0084-41bf-9063-8606453f222f.jpeg from the image_paths list
# for image_path in image_paths:
#     if image_path.name == "cd6bcc10-0084-41bf-9063-8606453f222f.jpeg":
#         image_paths.remove(image_path)

# image_paths = list(Path(images_dir).rglob("*.jp*g"))
# print(f"[INFO] Number of images after removal: {len(image_paths)}")

# Select a random subset
import random 
# image_subset = random.sample(num_images, 10000)
# print(f"[INFO] Number of images in subset: {len(image_subset)}")

# Copy each image to its own folder in temp
# # As in, temp/image_name/image_name.jpg
# import shutil
# temp_dir = "./notebooks/artifacts/temp/"

# # Delete temp dir if it exists
# if os.path.exists(temp_dir):
#     shutil.rmtree(temp_dir)

# # Make the temp dir
# os.makedirs(temp_dir, exist_ok=True)

# for image in tqdm(image_paths):
#     image_name = image.name
#     image_dir = os.path.join(temp_dir, image_name)
#     os.makedirs(image_dir, exist_ok=True)
#     shutil.copy2(image, image_dir)
#     # shutil.copy2(image, temp_dir)

# # Get a list of all the image folders in temp
# image_folders = list(Path(temp_dir).rglob("*"))

# image_paths = list(Path(temp_dir).rglob("*.jp*g"))

# Create output directory
output_dir = "duplicates" # TODO: make this arg
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":

    fd = fastdup.create(work_dir=output_dir,
                        input_dir=images_dir)

    fd.run(num_threads=4,
        threshold=0.96, # make this arg
        compute="cpu",
        verbose=True,
        run_stats=0,
        num_images=20000)
    
    # for image_folder in tqdm(image_folders):
    #     # print(image_folder)
    #     # Find duplicates
    #     print(f"[INFO] Finding duplicates in: {image_folder}")
    #     try: 
    #         fd = fastdup.create(work_dir=output_dir,
    #                             input_dir=image_folder)

    #         fd.run(num_threads=1,
    #             threshold=0.96, # make this arg
    #             compute="cpu",
    #             verbose=True,
    #             run_stats=0)

    #         successful_image_folders.append(image_folder)

    #     except Exception as e:
    #         print(e)
    #         failed_image_folders.append(image_folder)
    #         continue

    # print(f"[INFO] Successful image folders: {len(successful_image_folders)}")
    # print(f"[INFO] Failed image folders: {len(failed_image_folders)}")
    # print(f"[INFO] Failed image folders: {failed_image_folders}")

    # # Save failed image folders to file
    # with open("failed_image_folders.txt", "w") as f:
    #     for image_folder in failed_image_folders:
    #         f.write(f"{image_folder}\n")