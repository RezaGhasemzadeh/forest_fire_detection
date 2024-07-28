import cv2 
from sklearn.model_selection import train_test_split
import glob

data = []


def preprocessing(add):
    img = cv2.imread(add)
    img = cv2.resize(img, (32, 32))
    # Normalizing each pixel.
    img = img/255.0
    # Transforming the matrix into a feature vector(each pixel is a feature)
    img = img.flatten()
    return img


for address in glob.glob("fire_dataset\\*\\*"):
    image = preprocessing(address)
    data.append(image)
     
