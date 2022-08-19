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
    words_list = segmentor.words_list
    st.write(f"Ada {len(words_list)} kata di gambar.")
    st.image(segmented_image, caption='Your image')