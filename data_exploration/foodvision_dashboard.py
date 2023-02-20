"""
Streamlit dashboard for exploring data from FoodVision.

Basing off this: https://blog.streamlit.io/how-to-build-a-real-time-live-dashboard-with-streamlit/ 
"""
import pandas as pd
import streamlit as st
import altair as alt

# Import the data
# TODO: change this to a URL that gets live tracked?/updated etc
dataset_url = "annotations.csv"

# TODO: cache this so it's saved: https://docs.streamlit.io/library/advanced-features/caching 
def get_data() -> pd.DataFrame:
    """Get the data from a CSV file.
    """
    return pd.read_csv(dataset_url)

df = get_data()

# Setup dashboard
st.set_page_config(
    page_title="FoodVision Image Classification Dashboard 🍔👁️",
    page_icon="🍔",
    layout="wide",
)

st.title("FoodVision Image Classification Dashboard")

st.dataframe(df)

st.write("Number of rows:", len(df))

col1, col2 = st.columns(2)
col1.metric("Num rows", len(df))
col2.metric("Num classes", len(df.class_name.unique()))
# col3.metric("Humidity", "86%", "4%")

# Sort the class counts in descending order
class_counts_sorted = df.class_name.value_counts().reset_index().rename(columns={"index": "class_name", "class_name": "counts"})
st.subheader(f"Number of images per class")
class_counts_chart = alt.Chart(class_counts_sorted).mark_bar().encode(
    x=alt.X('class_name', sort=alt.EncodingSortField(field="counts", order="descending")),
    y='counts',
)
st.altair_chart(class_counts_chart, use_container_width=True)

# Create a series of columns for image_source per class
class_image_source_counts = df.groupby(["class_name", "image_source"]).size().reset_index(name="counts")

# Create a select option for different image_source values
st.subheader(f"Number of images per class for different image sources")
image_source = st.selectbox("Image source", class_image_source_counts.image_source.unique())

# Filter the class_image_source_counts to only include the selected image_source
class_image_source_counts_filtered = class_image_source_counts[class_image_source_counts.image_source == image_source][["class_name", "counts"]]

source_counts_chart = alt.Chart(class_image_source_counts_filtered).mark_bar().encode(
    x=alt.X('class_name', sort=alt.EncodingSortField(field="counts", order="descending")),
    y='counts',
)
st.altair_chart(source_counts_chart, use_container_width=True)

# Filter for only Harris Farm classes
harris_farm_classes = ['eggs', 'apple_green', 'apple_red', 'apricot', 'avocado', 'banana', 'blackberries', 'blueberries', 'cherries', 'coconut', 'dates', 'dragonfruit', 'figs', 'grapefruit', 'grapes_black', 'grapes_red', 'grapes_white', 'honeydew_melon', 'kiwifruit', 'lemon', 'lime', 'lychee', 'mandarin', 'mango', 'nectarine', 'orange', 'papaya', 'passionfruit', 'pawpaw', 'peach', 'pear', 'pear_nashi', 'persimmon', 'pineapple', 'plum', 'pomegranate', 'quince', 'raspberries', 'rockmelon', 'star_fruit', 'strawberries', 'watermelon', 'basil', 'bay_leaves', 'chervil', 'chives', 'coriander', 'curry_leaves', 'dill', 'ginger', 'lemongrass', 'lime_leaves', 'marjoram', 'mint', 'oregano', 'parsley', 'rosemary', 'sage', 'tarragon', 'thyme', 'turmeric', 'almonds', 'cashews', 'peanuts', 'walnuts', 'autumn', 'spring', 'summer', 'winter', 'artichoke', 'asparagus', 'bean_sprouts', 'beetroot', 'bok_choy', 'broccoli', 'broccolini', 'brussel_sprout', 'cabbage_green', 'cabbage_red', 'capsicum_green', 'capsicum_orange', 'capsicum_red', 'capsicum_yellow', 'carrot', 'cauliflower', 'celery', 'chickory', 'chilli', 'choy_sum', 'corn', 'cucumber', 'eggplant', 'endive', 'enoki_mushrooms', 'fennel', 'garlic', 'green_beans', 'jalapeno', 'kale', 'leek', 'lettuce_cos', 'lettuce_iceberg', 'mushrooms', 'okra', 'onion_brown', 'onion_green', 'onion_red', 'onion_white', 'parsnip', 'potato_brown', 'potato_red', 'potato_white', 'pumpkin', 'radish', 'rhubarb', 'shallots', 'silverbeet', 'snowpeas', 'spinach', 'squash', 'swede', 'sweet_potato', 'taroroot', 'tomato', 'turnip', 'watercress', 'witlof', 'wombok', 'zucchini', 'coffee', 'ice_coffee']
st.subheader(f"Harris Farm classes")
# image_source = st.selectbox("Image source", class_image_source_counts.image_source.unique())

# Filter the class_image_source_counts_filtered to only include harris_farm_classes
class_image_source_counts_filtered_hf = class_image_source_counts_filtered[class_image_source_counts_filtered.class_name.isin(harris_farm_classes)]

hf_source_counts_chart = alt.Chart(class_image_source_counts_filtered_hf).mark_bar().encode(
    x=alt.X('class_name', sort=alt.EncodingSortField(field="counts", order="descending")),
    y='counts',
)
st.altair_chart(hf_source_counts_chart, use_container_width=True)

st.write(harris_farm_classes)
class_names = sorted(df.class_name.unique().tolist())
st.write(class_names)
# st.subheader()
st.subheader("Missing classes (classes in Harris Farm but not in FoodVision)")
st.write(sorted(list(set(harris_farm_classes) - set(class_names))))