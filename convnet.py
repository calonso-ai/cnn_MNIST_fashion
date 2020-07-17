#Step 1: Data initialization and loading

import numpy as np
import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, MaxPooling2D, BatchNormalization
from keras import optimizers
import matplotlib.pyplot as plt

#MNIST data is downloaded and splitted in train/test set
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = fashion_mnist.load_data()

#First dataset image is visualized
first_image = x_train_orig[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

#Reshape and OneHotEncoder
x_train = x_train_orig.reshape(60000,28,28,1)
x_test = x_test_orig.reshape(10000,28,28,1)

y_train = to_categorical(y_train_orig, 10)
y_test = to_categorical(y_test_orig, 10)

#Step 2: Deep CNN + Dropout model creation, compile, training and evaluation

#A deep convolutional neural network will be created with below parameter:

#-Two convolution layers of 32 and 64 kernels, size 3x3 and stride 1, both with ReLU activation.
#-A maxpooling layer with a kernel size (2x2) and stride 2.
#-A dropout layer (with probability equal to 0.25).
#-A flatten layer.
#-A dense layer of 128 neurons with reluctance activation.
#-A dropout layer (with probability equal to 0.5).
#-A dense layer with softmax activation.

#The create_cnn function define a convolutional neural network with corresponding layers.
def create_cnn ():
    model = Sequential ()
    #Since the first layer of this cnn is a 32 kernel convolution layer, it is passed as an argument the
    # input_shape parameter, indicating that the input images will be 28x28 dimension and grayscale
    model.add (Conv2D (32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    # The second convolutional layer is created, in this case with 64 kernels
    model.add (Conv2D (64, (3, 3), activation = 'relu'))
    #Next, a two-dimensional maxpooling layer is created, indicating a stride = 2
    model.add (MaxPooling2D ((2, 2), strides = 2))
    #In this stage we add a Dropout layer that randomly removes a set of neurons from the stage
    #of training
    model.add (Dropout (0.25))
    #The flatten layer transforms a two-dimensional matrix (in this case, our images) into a vector that
    #can be processed by FC layer (fully connected)
    model.add (Flatten ())
    # We added a dense layer of 128 neurons with relu activation to interpret the characteristics.
    model.add (Dense (128, activation = 'relu'))
    model.add (Dropout (0.5))
    #Finally, we add an output layer with softmax activation function with 10 nodes, since the object of this
    #cnn is the classification among 10 kinds of objects.
    model.add (Dense (10, activation = 'softmax'))
    return model

#For compilation we will use:
#keras.losses.categorical_crossentropy as a function of loss
#keras.optimizers.Adadelta () as optimizacor
#The metric to optimize will be the accuracy
#To train the model we will use a batch_size = 128, epochs = 12 and the test set to validate.

# First, we create the model through the create_cnn function defined above
model = create_cnn ()
#Adadelta optimizer is defined. It is recommended to leave the default parameters in this optimizer.
sgd = optimizers.Adadelta (learning_rate = 1.0, rho = 0.95)
model.compile (loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Next, the model is trained using the test set as a validation set
hist_model = model.fit (x_train, y_train, validation_data = (x_test, y_test), epochs = 12, batch_size = 128)

#Finally, we use the test set to evaluate the model
_, acc = model.evaluate (x_test, y_test, verbose = 0)
print ('%. 3f'% (acc * 100.0))
 
