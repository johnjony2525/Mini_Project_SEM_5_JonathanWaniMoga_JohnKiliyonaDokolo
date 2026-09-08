import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = r"C:\PneumoniaAI\models\transfer_best.keras"
IMG_SIZE = (224, 224)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PneumoniaAI",
    page_icon="🫁",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()

# ==========================================
# TITLE
# ==========================================

st.title("🫁 PneumoniaAI")

st.subheader(
    "AI-Based Chest X-Ray Pneumonia Detection"
)

st.write(
    "Upload a chest X-ray image to receive an AI-based prediction."
)

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        # Resize image
        resized_image = image.resize(
            IMG_SIZE
        )

        # Convert image to array
        image_array = np.array(
            resized_image
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Make prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )[0][0]

        # Interpret prediction
        if prediction >= 0.5:

            result = "PNEUMONIA"
            confidence = prediction

        else:

            result = "NORMAL"
            confidence = 1 - prediction

        # Display result
        st.divider()

        st.subheader("Prediction Result")

        st.write(
            f"### {result}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )

        # Explanation
        st.info(
            ""
            "medical diagnosis."
        )