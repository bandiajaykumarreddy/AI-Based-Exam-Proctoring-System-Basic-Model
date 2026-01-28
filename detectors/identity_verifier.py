import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

embedder = FaceNet()

def get_embedding(face_rgb):
    face_rgb = face_rgb.reshape(1, 160, 160, 3)
    return embedder.embeddings(face_rgb)[0]

def average_embeddings(embeddings):
    return np.mean(embeddings, axis=0)

def is_impostor(baseline, current, threshold=0.45):
    dist = cosine(baseline, current)
    return dist > threshold