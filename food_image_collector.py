import streamlit as st
import datetime

from PIL import Image
from save_to_gsheets import append_values_to_gsheet
from utils import create_unique_filename, upload_blob
from rich import pretty, print, traceback 
pretty.install()
traceback.install()

st.title("Nutrify Image Collection 🍔👁")
st.write("Upload or take a photo of your food and help build the world's biggest food image database!")

uploaded_image = st.file_uploader(label="Upload an image of food",
                                  type=["png", "jpeg", "jpg"])

def display_image(img):
    if img is not None: 
        # Show the image
        img = Image.open(img)
        print("Displaying image...")
        print(img.height, img.width)
        st.image(img, width=400)
    return img

image = display_image(uploaded_image)

# Create image label form to submit
form = st.form(key="label_submit_form", clear_on_submit=True)
label = form.text_input(label="What food(s) are in the image you uploaded? \
    You can enter text like: 'ramen' or 'eggs, bread, bacon'",
    max_chars=200)
st.markdown('**Note:** If you click "upload image", your image will be stored on \
          Nutrify servers and used to create the largest food image database \
          in the world! *(Do not upload anything sensitive, as it may one day \
              become publicly available)*')
submit_button = form.form_submit_button(label="Upload image",
    help="Click to upload your image and label to Nutrify servers"
)

if submit_button:
    # Generate unique filename for the image
    unique_image_id = create_unique_filename()

    # Make timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Upload image object to Google Storage
    upload_blob(source_file_name=uploaded_image,
                destination_blob_name= unique_image_id + ".jpeg"
    )

    # Add image metadata to Gsheet
    img_height = image.height
    img_width = image.width
    image_info = [[unique_image_id, current_time, img_height, img_width, label]]
    response = append_values_to_gsheet(values_to_add=image_info)

    # Output details
    print(response)
    print(image)

st.markdown("View the source code for this page on [GitHub](https://github.com/mrdbourke/nutrify).")