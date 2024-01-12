import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import joblib


def extract_features(images):
    features = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, hog_features = hog(gray_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        features.append(hog_features.flatten())  # Aplatir les caractéristiques HOG

    return np.array(features)



def load_data(base_folder):
    images = []
    labels = []
    label_mapping = {}

    label_counter = 0
    for person_folder in os.listdir(base_folder):
        person_path = os.path.join(base_folder, person_folder)

        if not os.path.isdir(person_path):
            continue

        label_mapping[person_folder] = label_counter

        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                labels.append(label_counter)

        label_counter += 1

    return np.array(images), np.array(labels), label_mapping

if __name__ == "__main__":
    base_folder = 'C://Users//aitichou//Desktop//HOG//dataset'

    # Chargez les données
    images, labels, label_mapping = load_data(base_folder)

    # Divisez les données en ensembles de formation et de test
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Entraînez le modèle
    svm_model = SVC(kernel='linear')
    svm_model.fit(extract_features(X_train), y_train)

    # Évaluation du modèle sur l'ensemble de test
    y_pred = svm_model.predict(extract_features(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy}")

    # Enregistrez le modèle formé
    joblib.dump(svm_model, 'svm_model.joblib')
