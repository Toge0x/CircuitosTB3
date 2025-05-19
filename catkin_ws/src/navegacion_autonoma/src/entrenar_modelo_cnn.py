import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from modelo_cnn import build_cnn_model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === Parámetros ===
IMAGE_FOLDER = 'dataset/images'
LABEL_FILE = 'dataset/labels.csv'
IMG_SIZE = (64, 64)
EPOCHS = 25
BATCH_SIZE = 32

# === Cargar etiquetas ===
df = pd.read_csv(LABEL_FILE)

X = []
y = []

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row['filename'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    X.append(img)
    y.append([row['v'], row['w']])

X = np.array(X)
y = np.array(y)

# === División entrenamiento/validación ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

# === Construcción del modelo ===
model = build_cnn_model(input_shape=(64, 64, 3))
model.compile(optimizer=Adam(1e-4), loss='mse')

# === Entrenamiento ===
checkpoint = ModelCheckpoint('modelo_entrenado.h5', save_best_only=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])
