import streamlit as st
import numpy as np
import cv2
import numpy as np

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
uploaded_file = st.file_uploader(label="Choose a file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    segmentor = WordSegmentation()
    segmented_image = segmentor.segmentation(image=image)
    st.image(segmented_image, caption='Your image')