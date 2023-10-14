import streamlit as st
from feature_extraction import feature_extraction
import pickle5 as pickle
import joblib
import cv2
from sklearn.ensemble import RandomForestClassifier

model = pickle.load(open('RF_Rhea.pkl','rb'))

from PIL import Image

def main():
    # Set the page to have a white background
    st.set_page_config(page_title="Satellite Image Classification", layout="wide", page_icon=":earth_africa:")

    # Set a title and description
    st.title('Satellite Image Classification')
    st.write("Upload the satellite image")

    # Create an uploader widget for image upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "tif"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        image = Image.open(uploaded_image)
        temp_file = "temp_image.tif"  # You can use any supported image format
        image.save(temp_file)

        # Read the image using OpenCV
        image_cv2 = cv2.imread(temp_file)
        feature_vector = feature_extraction(image_cv2)

        prediction = model.predict(feature_vector)
        

        # Print the feature vector
        st.write("predic")
        st.write(prediction)

if __name__ == '__main__':
    main()