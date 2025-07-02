import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

import os
import shutil

# Paths
fashion_base = "/kaggle/input/deep-fashion-dataset-40k"
output_path = "/kaggle/working/latest_40k"
os.makedirs(output_path, exist_ok=True)

# Walk through all subdirectories and collect images
for root, dirs, files in os.walk(fashion_base):
    for fname in files:
        src = os.path.join(root, fname)
        # Add folder name prefix to avoid duplicate names
        folder_prefix = os.path.basename(root)
        dst = os.path.join(output_path, f"{folder_prefix}_{fname}")
        shutil.copy(src, dst)

print("✅ All images collected in:", output_path)

file_names = [os.path.join(output_path, fname)
              for fname in os.listdir(output_path)
              if os.path.isfile(os.path.join(output_path, fname))]


def extract_features_batch(img_paths, model, batch_size=32):
    features = []
    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_paths = img_paths[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            img = image.load_img(path, target_size=(224, 224))
            img = image.img_to_array(img)
            batch_images.append(img)

        batch_images = np.array(batch_images)
        batch_images = preprocess_input(batch_images)

        batch_features = model.predict(batch_images, verbose=0)

        # normalize each feature
        for feature in batch_features:
            features.append(feature / norm(feature))

    return np.array(features)


import pickle

features_array = extract_features_batch(file_names, model, batch_size=32)

pickle.dump(features_array, open('embeddings.pkl', 'wb'))
pickle.dump(file_names, open('filenames.pkl', 'wb'))

Now
modify
this
code
to
ViT?