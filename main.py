import cv2 
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump


def preprocessing(add):
    img = cv2.imread(add)
    img = cv2.resize(img, (32, 32))
    # Normalizing each pixel.
    img = img/255.0
    # Transforming the matrix into a feature vector(each pixel is a feature)
    img = img.flatten()
    label = add.split('\\')[2].split('.')[0]
    return img, label


def load_data():
    data = []
    labels = []
    for (i, address) in enumerate(glob.glob("fire_dataset\\*\\*")):
        image, label = preprocessing(address)
        data.append(image)
        labels.append(label)
        if i % 100 == 0:
            print(f"{i}/998 processed")

    # Because sklearn only accepts numpy arrays or pandas dataframes, we convert our list to 
    # a numpy array but there is no need to convert labels to array
    data = np.array(data)
    train_features, test_features, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    return train_features, test_features, train_labels, test_labels


def train():
    train_features, test_features, train_labels, test_labels = load_data()
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(train_features, train_labels)
    dump(knn_classifier, "fire_detection.z")
    accuracy = knn_classifier.score(test_features, test_labels)
    print(f"accuracy: {accuracy*100}")


train()

