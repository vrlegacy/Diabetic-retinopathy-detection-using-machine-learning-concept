import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to dataset
DATA_DIR = "data/images"
LABEL_FILE = "data/labels.csv"  # CSV with columns: filename,label

IMG_SIZE = 128  # Resize images

def load_data():
    df = pd.read_csv(LABEL_FILE)
    images = []
    labels = []

    for index, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(row['label'])

    images = np.array(images) / 255.0  # Normalize
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    print("Data preprocessing completed and saved.")

