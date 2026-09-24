#!pip install streamlit

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

import pickle
import os
import cv2
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
from PIL import Image
from mtcnn import MTCNN


# -----------------------------
# LOAD MODEL & DATA
# -----------------------------

detector = MTCNN()

model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)

feature_list = np.array(
    pickle.load(open('embedding.pkl', 'rb'))
)

filenames = pickle.load(
    open('filenames.pkl', 'rb')
)


# -----------------------------
# SAVE UPLOADED IMAGE
# -----------------------------

def save_uploaded_image(uploaded_image):

    try:

        os.makedirs('uploads', exist_ok=True)

        with open(
            os.path.join('uploads', uploaded_image.name),
            'wb'
        ) as f:

            f.write(uploaded_image.getbuffer())

        return True

    except Exception as e:

        st.error(str(e))
        return False


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_features(img_path, model, detector):

    img = cv2.imread(img_path)

    if img is None:
        return None

    # Detect face
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    # Get largest face
    result = max(
        results,
        key=lambda x: x['box'][2] * x['box'][3]
    )

    x, y, width, height = result['box']

    # Prevent negative coordinates
    x = max(0, x)
    y = max(0, y)

    face = img[
        y:y + height,
        x:x + width
    ]

    # BGR → RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = Image.fromarray(face)

    image = image.resize(
        (224, 224)
    )

    # NumPy
    face_array = np.asarray(
        image
    ).astype('float32')

    # Add batch dimension
    expanded_img = np.expand_dims(
        face_array,
        axis=0
    )

    # VGGFace preprocessing
    preprocessed_img = preprocess_input(
        expanded_img
    )

    # Feature extraction
    result = model.predict(
        preprocessed_img,
        verbose=0
    ).flatten()

    return result


# -----------------------------
# RECOMMEND
# -----------------------------

def recommend(feature_list, features):

    similarity = cosine_similarity(
        features.reshape(1, -1),
        feature_list
    )[0]

    index_pos = np.argmax(similarity)

    return index_pos, similarity[index_pos]


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title(
    'Which Bollywood Celebrity Are You?'
)

uploaded_image = st.file_uploader(
    'Choose an image',
    type=['jpg', 'jpeg', 'png']
)


if uploaded_image is not None:

    # Save image
    if save_uploaded_image(uploaded_image):

        image_path = os.path.join(
            'uploads',
            uploaded_image.name
        )

        # Display uploaded image
        display_image = Image.open(
            uploaded_image
        )

        # Extract features
        features = extract_features(
            image_path,
            model,
            detector
        )

        if features is None:

            st.error(
                "No face detected. "
                "Please upload a clear face image."
            )

        else:

            # Prediction
            index_pos, score = recommend(
                feature_list,
                features
            )

            # Convert Windows path to Linux-compatible path
            result_image_path = filenames[index_pos].replace('\\', '/')

            # Celebrity name
            predicted_actor = os.path.basename(
                os.path.dirname(result_image_path)
            ).replace('_', ' ')

            # Display
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your uploaded image')
                st.image(
                    display_image
                )

            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(
                    result_image_path,
                    width=300
                )
                st.write(f"Similarity: {score:.2%}")