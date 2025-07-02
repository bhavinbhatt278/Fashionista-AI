import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# === Load embeddings chunks from local folder ===
LOCAL_EMBEDDINGS_FOLDER = "embeddings_chunks"  # Update with your local folder path

chunk_files = sorted([
    os.path.join(LOCAL_EMBEDDINGS_FOLDER, f)
    for f in os.listdir(LOCAL_EMBEDDINGS_FOLDER)
    if f.startswith("embeddings_chunk_") and f.endswith(".pkl")
])

embedding_chunks = []
for chunk_file in chunk_files:
    with open(chunk_file, "rb") as f:
        chunk = pickle.load(f)
        embedding_chunks.append(chunk)

feature_list = np.concatenate(embedding_chunks, axis=0)

# === Load filenames locally ===
with open("filenames_aws.pkl", "rb") as f:
    filenames = pickle.load(f)

# === Load model ===
model_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_base.trainable = False
model = tensorflow.keras.Sequential([
    model_base,
    GlobalMaxPooling2D()
])

# === Streamlit UI ===
st.title('🛍️ Fashion Recommender System')

if "cart" not in st.session_state:
    st.session_state.cart = []

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

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
                image_url = filenames[idx]  # S3 URL or local path if you updated filenames accordingly
                st.image(image_url, use_container_width=True)

                raw_name = os.path.basename(image_url)
                clean_name = raw_name.split('_img_')[0].replace('_', ' ').strip()

                if st.button(f"Add to Cart {clean_name}", key=idx):
                    st.session_state.cart.append(image_url)
                    st.success(f"Added {clean_name} to cart ✅")
    else:
        st.header("Some error occurred in file upload")

# --- Sidebar Cart ---
st.sidebar.header("🛒 Your Cart")
if st.session_state.cart:
    for item in st.session_state.cart:
        st.sidebar.image(item, width=100)
        filename = os.path.basename(item)
        clean_name = filename.split('_img_')[0].replace('_', ' ').strip()
        st.sidebar.text(clean_name)
else:
    st.sidebar.text("Your cart is empty.")

if st.sidebar.button("Proceed to Checkout"):
    st.sidebar.success("✅ Checkout successful! Order will be shipped soon")
    st.session_state.cart = []
