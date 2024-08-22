import cv2
import glob
from joblib import load
import numpy as np

model = load("fire_detection.z")

for item in glob.glob("test_images\\*"):
    image = cv2.imread(item)
    image_resized = cv2.resize(image, (32, 32))
    image_resized = image_resized/255.0
    image_resized = image_resized.flatten()
    image_resized = np.array([image_resized])
    predicted_label = model.predict(image_resized)[0]
    cv2.putText(image, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (255, 0, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()