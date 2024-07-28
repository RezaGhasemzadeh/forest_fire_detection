import cv2 
import glob
from sklearn.model_selection import train_test_split

data = []
labels = []


def preprocessing(add):
    img = cv2.imread(add)
    img = cv2.resize(img, (32, 32))
    # Normalizing each pixel.
    img = img/255.0
    # Transforming the matrix into a feature vector(each pixel is a feature)
    img = img.flatten()
    label = add.split('\\')[2].split('.')[0]
    return img, label


for address in glob.glob("fire_dataset\\*\\*"):
    image, label = preprocessing(address)
    data.append(image)
    labels.append(label)
     
