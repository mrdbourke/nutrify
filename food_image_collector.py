import streamlit as st
import datetime

from PIL import Image

from save_to_gsheets import append_values_to_gsheet
from utils import create_unique_filename, upload_blob
from rich import pretty, print, traceback

pretty.install()
traceback.install()

st.title("Nutrify Image Collection 🍔👁")
st.write(
    "Upload or take a photo of your food and help build the world's biggest food image database!"
)

uploaded_image = st.file_uploader(
    label="Upload an image of food",
    type=["png", "jpeg", "jpg"],
    help="Tip: if you're on a mobile device you can also take a photo",
)


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
st.write("## Image details")
with st.form(key="image_metadata_submit_form", clear_on_submit=True):
    # Image label
    label = st.text_input(
        label="What food(s) are in the image you uploaded? \
        You can enter text like: 'ramen' or 'eggs, bread, bacon'",
        max_chars=200,
    )

    # Image upload location
    country = st.text_input(
        label="Where are you uploading this delicious-looking food image from?",
        value="US",
        autocomplete="country",
    )
    st.caption(
        "Country level is fine, for example 'AU' for Australia or 'IND' for India"
    )

    # Person email
    email = st.text_input(
        label="What's your email? (optional, we'll use this to contact you about the app/say thank you for your image(s))",
        autocomplete="email",
    )

    # Disclaimer
    st.info(
        '**Note:** If you click "upload image", your image will be stored on \
            Nutrify servers and used to create the largest food image database \
            in the world! *(Do not upload anything sensitive, as it may one day \
                become publicly available)*'
    )

    # Submit button + logic
    submit_button = st.form_submit_button(
        label="Upload image",
        help="Click to upload your image and label to Nutrify servers",
    )

    if submit_button:
        if uploaded_image is None:
            st.error("Please upload an image")
        else:
            # Generate unique filename for the image
            unique_image_id = create_unique_filename()

            # Make timestamp
            current_time = datetime.datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S"
            )

            # Upload image object to Google Storage
            with st.spinner("Sending your image across the internet..."):
                upload_blob(
                    source_file_name=uploaded_image,
                    destination_blob_name=unique_image_id + ".jpeg",
                )
            st.success(
                f"Your image of {label} has been uploaded! Thank you :)"
            )

            # Add image metadata to Gsheet
            img_height = image.height
            img_width = image.width
            image_info = [
                [
                    unique_image_id,
                    current_time,
                    img_height,
                    img_width,
                    label,
                    country,
                    email,
                ]
            ]
            response = append_values_to_gsheet(values_to_add=image_info)

            # Output details
            print(response)
            print(image)

st.write("## FAQ")
with st.expander("What happens to my image?"):
    st.write(
        """
    When you click "upload image", your image gets stored on Nutrify servers (a big hard drive on Google Cloud).\n
    Here's a pretty picture which describes it in more detail:
    """
    )
    st.image("./images/image-uploading-workflow.png")
    st.write(
        "Later on, images in the database will be used to train a computer vision model to power Nutrify."
    )
with st.expander("Why do you need images of food?"):
    st.write(
        """
    Machine learning models learn by looking at many different examples of things.\n
    Food included.\n
    Eventually, Nutrify wants to be an app you can use to *take a photo of food and learn about it*.\n
    To do so, we'll need many different examples of foods to build a computer vision
    model capable of identifying almost anything you can eat.\n
    And the more images of food you upload, the better the models will get.
    """
    )

st.markdown(
    "View the source code for this page on [GitHub](https://github.com/mrdbourke/nutrify)."
)
