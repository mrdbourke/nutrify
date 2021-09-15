# Nutrify - take a photo of food and learn about it

Work in progress.

To make this a thing, we're going to need lots of food images...

Start uploading your food images here: https://share.streamlit.io/mrdbourke/nutrify/main/food_image_collector.py 

Streaming progress on [Twitch](https://www.twitch.tv/mrdbourke)/making videos about it on YouTube.

**End goal:** take a photo of food an learn about it (nutrition information, where it's from, recipes, etc).

Something like this (a data flywheel for food images):
![](images/food-vision-data-flywheel-v1.png)

**Status:** making a small application to collect large amounts of food images.

## What's in this repo?
* `images/` - folder with misc images for the project
* `food_image_collector.py` - Streamlit-powered app that collects photos and uploads them to a Google Storage bucket and stores metadata in Google Sheets (these are private), see the workflow below. 
* `save_to_gsheets.py` - Small utility script that saves a bunch of metadata about an uploaded image to a Google Sheet (this will likely move into a dedicated `utils/` folder later on.
* `utils.py` - Series of helper functions used in `food_image_collector.py`, for example, `upload_blod()`, a function that uploads a photo to Google Storage.
* `requirements.txt` - A text file with the dependency requirements for this project.

## Image uploading workflow

The script `food_image_collector.py` is currently hosted using Streamlit Cloud. It does this:

![](images/image-uploading-workflow.png)
