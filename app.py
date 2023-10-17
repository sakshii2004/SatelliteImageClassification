import streamlit as st
from feature_extraction_2 import extract_features
from preprocess import preprocess
import pickle
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import time

# Loading the pickled model
model = pickle.load(open('modelRF_Final.pkl', 'rb'))


def main():
    st.set_page_config(page_title="Urban Land Cover Classification",
                       layout="wide",
                       page_icon=":earth_africa:")

    st.title("Urban Land Cover Classification")

    with st.sidebar:
        st.title("About our project :earth_africa:")
        st.header("Urban Land Cover Classification with Random Forest")
        st.write(
            "Welcome to our Urban Land Cover Classification project! We've trained a Random Forest classifier on a dataset of 2,100 satellite images, categorized into 21 different classes. This model can predict the land cover of urban areas (currently with an accuracy of nearly 70%), offering valuable insights for urban planning and environmental studies. \n\n Explore our application, upload a satellite image, and let our model classify the land cover for you. We hope this tool can assist in urban development and environmental monitoring. \n\n Enjoy your exploration of urban land cover classification!"
        )

    uploaded_image = st.file_uploader(
        "Upload the satellite image",
        type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded_image is not None:

        st.image(uploaded_image, caption='Uploaded Image', width=200)

        image = Image.open(uploaded_image)
        temp_file = "temp_image.tif"
        image.save(temp_file)

        # Read the image using OpenCV
        image_cv2 = cv2.imread(temp_file)

        progress_bar1 = st.progress(0, "Extracting Features...")

        rgb_image, gray_image = preprocess([image_cv2])
        feature_vector = extract_features(rgb_image, gray_image)

        for percent_complete in range(100):
            time.sleep(0.015)
            progress_bar1.progress(percent_complete + 1,
                                   "Extracting Features...")

        time.sleep(0.7)
        progress_bar1.empty()
        st.write('Features extracted!')

        progress_bar2 = st.progress(0, "Predicting land cover class...")

        prediction = model.predict(feature_vector)

        class_dict = {
            0: 'Agricultural Land',
            1: 'Airplane',
            2: 'Baseball Diamond',
            3: 'Beach',
            4: 'Buildings',
            5: 'Chaparral',
            6: 'Dense Residential',
            7: 'Forest',
            8: 'Freeway (Roads)',
            9: 'Golf Course',
            10: 'Harbor',
            11: 'Intersection',
            12: 'Medium Residential',
            13: 'Mobile Home Park',
            14: 'Over Pass',
            15: 'Parking Lot',
            16: 'River',
            17: 'Runway',
            18: 'Sparse Residential',
            19: 'Storage Tanks',
            20: 'Tennis Court'
        }

        for percent_complete in range(100):
            time.sleep(0.015)
            progress_bar2.progress(percent_complete + 1,
                                   "Predicting land cover class...")

        time.sleep(0.7)
        progress_bar2.empty()
        st.write('Prediction completed!')

        prediction_class = class_dict[prediction[0]]
        st.success(f"Predicted class is : {prediction_class}")


if __name__ == '__main__':
    main()
