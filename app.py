## Dependencies
### Libraries
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import imutils
import torch
import cv2
import io
import os

### Functions
#### Crop Image Entry
#### = > def crop_image
#### Image MANIPULATION
def save_predictedfile(uploadedfile):
    with open(os.path.join("./content/", "brain-tumour.jpg"), "wb") as f:
        f.write(uploadedfile)

def save_uploadedfile(uploadedfile):
    with open(os.path.join("./content/", "selfie.jpg"), "wb") as f:
        f.write(uploadedfile.getbuffer())

def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    ## READ Image
    img = np.array(Image.open(img))

    ## CONVERT TO GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
              extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)

    new_img_dir = Image.fromarray(new_img)
    new_img_dir.save("./content/brain-tumour-cropped.jpg")

    return new_img

def add_margin(img):
    n_img = crop_img(img)
    cropped_image = Image.fromarray(n_img)
    top = right = bottom = left = 75
    width, height = cropped_image.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(cropped_image.mode, (new_width, new_height), 0)
    result.paste(cropped_image, (left, top))
    result.save('./content/brain-tumour-padded.jpg', quality=95, transparency=0)
    return result

def show_table(img):
    img = np.array(Image.open(img))
    result = model(img)
    prediction_table = result.pandas().xyxy[0]
    prediction_table = pd.DataFrame(prediction_table)

    top_n = prediction_table[prediction_table.index.isin([0])]

    predicted_class = top_n['name'][0]

    return top_n, predicted_class

def predict_img(img):
    # img = crop_img(img) # Apply-Augmentation

    image = np.array(Image.open(img))

    result = model(image)

    output = io.BytesIO()
    out_image = np.squeeze(result.render())
    output_img = Image.fromarray(out_image)
    output_img.save(output, format='JPEG')
    result_img = output.getvalue()

    save_predictedfile(result_img)

    return st.image(result_img)

## Load Model
run_model_path = 'last_20_22.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path = run_model_path)
model.eval()

DEMO_IMAGE = './content/demo.jpg'

print("LOADED MODEL")

## BUILDING BACKGROUND

## Application States
APPLICATION_MODE = st.sidebar.selectbox("Our Options",
                                        ["About the App", "Take a Selfie", "Predict"]
                                        )

## Introduction
st.title("BRAIN TUMOUR DETECTION")
st.subheader("An implementation of Machine Learning to determine the extent and class of brain tumour"
             " in scanned x-ray images")
st.markdown("---")

if st.checkbox("About the Problem"):
    st.subheader(
        "A Brain tumor is considered as one of the aggressive diseases, among children and adults\n"
        "Brain tumors account for "
        "85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed "
        "with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately"
        "34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, "
        "Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve"
        " the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI)."
        "A huge amount of image data is generated through the scans. These images are examined by the radiologist." 
        "A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties. \n\n"
        
        "Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has "
        "consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and "
        "classification by using Deep Learning Algorithms using ConvolutionNeural Network (CNN), Artificial Neural Network (ANN), "
        "and TransferLearning (TL) would be helpful to doctors all around the world. \n\n"
        
        "Acknowledgements for Dataset.\n\n"
        "**Navoneel Chakrabarty** & **Swati Kanchan**\n\n"
        
        "Powered by YOLOv5 and PyTorch! \n"
                 "\n"
                 "Author: **AfroLogicInsect**")

    st.markdown("---")

### APPLICATION MODES
if APPLICATION_MODE == "About the App":
    st.markdown("**Web Graphical User Interface** \n"
                "Follow the side bar options, "
                "take a selfie with your device if you do not have an image to upload, \n"
                "Predict, to test our predictions \n")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false] > div:first-child{
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        Share your feedback with me - danielakan71@gmail.com
        """
    )

if APPLICATION_MODE == "Take a Selfie":
    picture = st.camera_input("Take a picture")

    if picture:
        st.sidebar.image(picture, caption="Selfie")
        if st.button("Save Image"):
            ## Function to save image
            save_uploadedfile(picture)
            st.sidebar.success("Saved File - Click to Download")
            selfie_img = "./content/selfie.jpg"
            with open(selfie_img, "rb") as file:
                btn = st.sidebar.download_button(
                    label="Download",
                    data=file,
                    file_name="selfie.jpg",
                    mime="image/jpeg")

    st.write("Click on **Clear photo** to retake picture")

elif APPLICATION_MODE == "Predict":
    st.sidebar.write(
        """
            Brain Tumour is one of the ravaging health deficiency that affects both the young and old.\n
            Like every other disease, early diagnosis presents a better chance of survival.
        """
    )
    
    st.sidebar.markdown("---")

    st.sidebar.write("**Use your own image**")
    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        UPLOADED_IMAGE = add_margin(img_file_buffer)
        UPLOADED_IMAGE = "./content/brain-tumour-padded.jpg"
        DEMO_IMAGE = UPLOADED_IMAGE
        try:
            p_table, p_class = show_table(UPLOADED_IMAGE)
            st.table(p_table)
            if p_class != "notumour":
                st.warning(f'Predicted class is  ***{p_class}***. Please see a Doctor')
            elif p_class == "notumour":
                st.success(f'Clean! Predicted class is  ***{p_class}***!')
        except:
            st.error("Unable to fetch a prediction at the moment, try with a larger image")

        if st.checkbox("Show Bounding Box"):
            predict_img(UPLOADED_IMAGE)

            ## Save Result
            result_img = "./content/brain-tumour.jpg"
            with open(result_img, "rb") as file:
                btn = st.download_button(
                    label="Save Result",
                    data=file,
                    file_name="brain-tumour.jpg",
                    mime="image/jpeg")

    elif st.sidebar.button("Use your Selfie"):
        SELFIE_IMAGE = "./content/selfie.jpg"
        add_margin(SELFIE_IMAGE)
        SELFIE_IMAGE = "./content/brain-tumour-padded.jpg"
        DEMO_IMAGE = SELFIE_IMAGE
        predict_img(SELFIE_IMAGE)

        try:
            p_table, p_class = show_table(SELFIE_IMAGE)
            st.table(p_table)
            if p_class != "notumour":
                st.warning(f'Predicted class is  ***{p_class}***. Please see a Doctor')
            elif p_class == "notumour":
                st.success(f'Clean! Predicted class is  ***{p_class}***!')
        except:
            st.error("Unable to fetch a prediction at the moment, try with a larger image")

        st.sidebar.markdown("---")
        st.markdown("---")

    else:
        DEMO_IMAGE = './content/demo.jpg'
        predict_img(DEMO_IMAGE)
        try:
            p_table, p_class = show_table(DEMO_IMAGE)
            st.table(p_table)
            st.success(f'Predicted class is  ***{p_class}***.')
        except:
            st.error("Unable to fetch a prediction at the moment. Try again with a larger image")

    st.markdown("---")

    st.caption("N: Predictions are not absolute, professional guidance is advised.")

    ## Place Demo
    st.sidebar.text("Placed Image")
    st.sidebar.image(DEMO_IMAGE)

###


