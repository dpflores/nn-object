# Lets predict other data that is not in our training

import cv2
import numpy as np
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

# Read the image as grayscale as our training data (doing the preprocessing)
image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)

# Resize to 28x28 resolution as the training data
image_data = cv2.resize(image_data, (28, 28))

# As we can see, the grayscaled that we used is inverted (background black instead of white)
image_data = 255 - image_data

import matplotlib.pyplot as plt
plt.imshow(image_data, cmap='gray')
# OpenCV uses BGR whil matplotlib RGB, so cvtColor
plt.show()

# Reshape the data and scalin -1 to 1 (remeber this is 1 sample)
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Now load the model to predict the data

# Load the model
model = Model.load('fashion_mnist.model')

# Predict on the image
confidences = model.predict(image_data)

# Get predictions instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
predictions = fashion_mnist_labels[predictions[0]]

print(predictions)

# Now we did it with a different type of data (you can try new data from pictures
# you took and try it again)

