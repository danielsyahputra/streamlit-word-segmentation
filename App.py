import streamlit as st
import numpy as np
import numpy as np
from PIL import Image

from utils import WordSegmentation

# Page settings
st.set_page_config(
    page_title="Word Segmentation App",
    layout="wide",
    initial_sidebar_state="expanded"
 )

# Title
st.title('Text-Based Image Segmentation')

# Upload file
uploaded_file = st.file_uploader(label="Choose a file", type=['jpg', 'jpeg'])

sidebar = st.sidebar

with open("assets/tulisan.jpg", "rb") as file:
    btn = sidebar.download_button(
            label="Download sample image",
            data=file,
            file_name="sample.jpg",
            mime="image/jpg"
        )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    segmentor = WordSegmentation()
    segmented_image = segmentor.segmentation(image=image)
    num_words = len(segmentor.words_list)

    # Show result
    show_original_image = sidebar.checkbox(label="Show original image", value=True)
    n_th = sidebar.number_input(label="Get nth word", min_value=0, max_value=(num_words - 1))
    n_th_word = segmentor.get_nth_word(n=n_th)
    sidebar.image(n_th_word, caption=f"Word - {n_th}")

    col1, col2 = st.columns([0.5, 0.5])

    if show_original_image:
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.image(image, width=425)
        with col2:
            st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
            st.image(segmented_image, width=425)
    else:
        st.image(segmented_image, caption="Segmented Image")