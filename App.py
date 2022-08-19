from distutils.command.upload import upload
import streamlit as st
import numpy as np
import cv2
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
st.title('Word Segmentation App')

# Upload file
uploaded_file = st.file_uploader(label="Choose a file", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    segmentor = WordSegmentation()
    segmented_image = segmentor.segmentation(image=image)
    num_words = len(segmentor.words_list)

    # Show result
    show_segmented_image = st.checkbox(label="Show segmented image", value=True)
    n_th = st.number_input(label="Get nth word", min_value=0, max_value=(num_words - 1))
    n_th_word = segmentor.get_nth_word(n=n_th)
    st.image(n_th_word, caption=f"Word - {n_th}")
    if show_segmented_image:
        st.image(segmented_image, caption='Your image')