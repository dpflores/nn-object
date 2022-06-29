# Lets implement the prediction
import numpy as np
import cv2
import os
from mynnfs import *

# Data labels
fashion_mnist_labels = {
    0 : 'T-shirt/top' ,
    1 : 'Trouser' ,
    2 : 'Pullover' ,
    3 : 'Dress' ,
    4 : 'Coat' ,
    5 : 'Sandal' ,
    6 : 'Shirt' ,
    7 : 'Sneaker' ,
    8 : 'Bag' ,
    9 : 'Ankle boot'
}

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path,dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folders
    for label in labels:
        # And for each image in teh folders
        for file in os.listdir(os.path.join(path,dataset,label)):
            # Read the image
            image = cv2.imread(os.path.join(path,dataset,label, file), cv2.IMREAD_UNCHANGED)

            # And append it and the label to the lists
            X.append(image)
            y.append(label)
        
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data

    return X, y, X_test, y_test



# With these functions we can create the data using
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffling the data with indexes, to equally shuffle both data and labels
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]


# Scale and reshape samples
X = (X.reshape(X.shape[0],-1 ).astype(np.float32)-127.5)/127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5)/127.5

# 2 hidden layers using ReLU activation, an ouput layer with softmax activation.
# since it is classification model, cross-entropy loss, Adam optimizer and categorical accuracy.

# Instantiate the model
model = Model.load('fashion_mnist.model')

# Predict on the first 5 samples from validation dataset 
# and print the result
confidences = model.predict(X_test[:5]) # this gives me an approximation for each label

# We will use the predictions property of our Activation function
predictions = model.output_layer_activation.predictions(confidences)

for prediction in predictions:
    print(fashion_mnist_labels[prediction])

# We have done it
