import streamlit as st
import numpy as np
import cv2
from PIL import Image
from preprocessing.contour_cropper import crop_brain_contour

def render_contour_tab(eval_img: Image.Image):
    """
    Renders Tab 3: Contour Cropping & Morphological Skull Stripping Pipeline.
    Displays Gaussian filtering, Otsu binary mask, and isolated brain tissue crop.
    """
    st.markdown("### Contour-Based Skull Stripping Pipeline")
    st.markdown("""
    Standardizes MRI inputs by removing exterior non-brain artifacts (black margins, scanner labels, calvarial padding).
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Grayscale & Gaussian Blur**")
        st.caption("Noise reduction using a 5x5 kernel.")
        gray = cv2.cvtColor(np.array(eval_img), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        st.image(blurred, caption="Gaussian Filtered (5x5)", width='stretch')

    with c2:
        st.markdown("**2. Morphological Segmentation**")
        st.caption("Otsu binarization with erosion and dilation iterations.")
        _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        st.image(thresh, caption="Binary Brain Parenchyma Mask", width='stretch')

    with c3:
        st.markdown("**3. Bounding Box & Tissue Crop**")
        st.caption("Cropped directly along extreme tissue coordinates.")
        cropped_view = crop_brain_contour(np.array(eval_img))
        st.image(cropped_view, caption="Isolated Brain Tissue (Cropped)", width='stretch')
