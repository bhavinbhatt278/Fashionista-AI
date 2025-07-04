# Fashionista_AI
Fashionista_AI is an AI-powered women’s fashion recommender that helps users discover wardrobe inspirations similar to an uploaded image. By leveraging deep learning and computer vision techniques, the system extracts visual features from the user’s photo and retrieves the most visually similar clothing items from a curated apparel database.

![Image Description](https://github.com/bhavinbhatt278/Fashionista-AI/blob/main/Image_1.png)
![Image Description](https://github.com/bhavinbhatt278/Fashionista-AI/blob/main/Img_2.png)


## 🚀 Features
✅ Upload a photo of a desired apparel style
✅ Extracts ResNet-based embeddings of the uploaded image
✅ Compares the embedding to a database of 40,000 fashion images (DeepFashion dataset)
✅ Returns the top 6 most visually similar results
✅ Displays results with names and images in an intuitive Streamlit interface
✅ Backend accelerated with GPU-based ResNet feature extraction
✅ Uses Nearest Neighbors for efficient similarity search

## 🛠️ Tech Stack
* Python 3.9+
* Streamlit (frontend)
* TensorFlow / Keras (ResNet model)
* scikit-learn (Nearest Neighbors)
* NumPy
* AWS S3 for storage
* Kaggle GPU (T4x2) for initial embedding generation
* DeepFashion Dataset (for the 40,000 reference images)

## 🖼️ Dataset
DeepFashion dataset (40,000 images sampled)

https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

## 🚧 Deployment Challenges
Due to the 336MB size of the embeddings file, repeated downloads from an S3 bucket on a free-tier server caused data transfer limits to be exceeded. In its current state, the application works perfectly on local environments but needs a persistent or vector database-backed solution for production deployment.

## Demo
[![Watch the demo](https://youtu.be/LW-XN56iA1I)


## 📌 Known Limitations
Dependent on large embeddings file

Free-tier servers spin down and reload data, incurring bandwidth costs

Designed for demonstration; not yet integrated with a live e-commerce store




