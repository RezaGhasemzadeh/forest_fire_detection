# Fire!
This project is a binary classification task that categorizes images as either fire or non-fire. The model is designed to detect fire in jungles.

The method I used to classify images is somewhat obsolete. First, I resized every image to 32x32 pixels and then normalized each pixel value by dividing by 255.

Next, I used the flatten() method to convert each image into a feature vector, where every pixel is treated as a feature.

I employed the K-Nearest Neighbors (KNN) algorithm to calculate the Euclidean distance between samples. The algorithm selects the 5 nearest neighbors and makes a decision on whether the image depicts fire or not based on the majority vote.

One of the main issues with this method is that the spatial relationships between pixels are not taken into account, a problem that Convolutional Neural Networks (CNNs) address effectively.

Moreover, KNN might not be the best choice when dealing with large datasets with a substantial number of features due to the computational expense of calculating distances between samples (typically using Euclidean distance).  

In the use_model.py I loaded the model and tested it with some new images
