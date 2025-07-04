import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# -------------------------
# Load embeddings & model
# -------------------------
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames_updated.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# --------------------------
# Streamlit UI
# --------------------------
st.title('🛍️ Fashionista AI')

# Session-state cart
if "cart" not in st.session_state:
    st.session_state.cart = []

# Save upload
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# --------------------------
# Upload and show results
# --------------------------
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)

        cols = st.columns(5)
        for idx, col in zip(indices[0][0:5], cols):
            with col:
                st.image(filenames[idx], use_container_width=True)

                # === Clean name for "Add to Cart" button ===
                raw_name = os.path.basename(filenames[idx])
                clean_name = raw_name.split('_img_')[0].replace('_', ' ').strip()

                if st.button(f"Add to Cart {clean_name}", key=idx):
                    st.session_state.cart.append(filenames[idx])
                    st.success(f"Added {clean_name} to cart ✅")
    else:
        st.header("Some error occurred in file upload")

# --------------------------
# SIDEBAR CART
# --------------------------
st.sidebar.header("🛒 Your Cart")
if st.session_state.cart:
    for item in st.session_state.cart:
        st.sidebar.image(item, width=100)

        # === Clean name for sidebar ===
        filename = os.path.basename(item)
        clean_name = filename.split('_img_')[0].replace('_', ' ').strip()
        st.sidebar.text(clean_name)
else:
    st.sidebar.text("Your cart is empty.")

if st.sidebar.button("Proceed to Checkout"):
    st.sidebar.success("✅ Checkout successful! Order will  be shipped soon")
    st.session_state.cart = []  # Clear the cart

