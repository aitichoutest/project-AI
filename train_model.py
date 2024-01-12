import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
import joblib
from sklearn.model_selection import train_test_split
import os

def extract_features(images):
    features = []
    sample_feature_size = None

    for image in images:
        # Convertir l'image en niveaux de gris si elle est en couleur
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Redimensionner l'image pour avoir une taille fixe
        resized_image = cv2.resize(gray_image, (250, 250))

        _, hog_features = hog(resized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        
        # Initialiser la taille des caractéristiques pour la première image
        if sample_feature_size is None:
            sample_feature_size = hog_features.shape[0]

        # Redimensionner les caractéristiques si elles ne correspondent pas à la taille attendue
        if hog_features.shape[0] != sample_feature_size:
            hog_features = cv2.resize(hog_features, (sample_feature_size,))

        features.append(hog_features.flatten())  # Utiliser flatten() pour rendre les caractéristiques bidimensionnelles

    return np.array(features)



def train_and_save_model(X_train, y_train, model_filename='svm_model.joblib'):
    # Imprimer les dimensions des caractéristiques extraites avant l'entraînement
    extracted_features = extract_features(X_train)
    print(f"Dimensions des caractéristiques extraites: {extracted_features.shape}")

    svm_model = SVC(kernel='linear')
    print("Length of X_train:", len(X_train))
    print("Length of y_train:", len(y_train))
    
    # Utiliser les caractéristiques extraites pour l'entraînement
    svm_model.fit(extracted_features, y_train)

    # Enregistrez le modèle formé
    joblib.dump(svm_model, model_filename)
def load_images(base_folder):
    images = []
    labels = []
    label_mapping = {}

    label_counter = 0
    for person_folder in os.listdir(base_folder):
        person_path = os.path.join(base_folder, person_folder)

        if not os.path.isdir(person_path):
            continue

        label_mapping[person_folder] = label_counter
        person_samples = 0  # Ajout pour compter le nombre d'échantillons par personne

        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                # Convertir l'image en niveaux de gris si elle est en couleur
                if len(image.shape) == 3:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = image

                images.append(gray_image)
                labels.append(label_counter)
                person_samples += 1

        print(f"Person: {person_folder}, Samples: {person_samples}")
        label_counter += 1

    print(f"Total Labels: {len(labels)}")
    return images, labels


if __name__ == "__main__":
    base_folder = 'C://Users//aitichou//Desktop//HOG//dataset'
    images, labels = load_images(base_folder)
    print(f"Length of images: {len(images)}, Length of labels: {len(labels)}")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Appelez la fonction pour entraîner et enregistrer le modèle
    train_and_save_model(X_train, y_train)
