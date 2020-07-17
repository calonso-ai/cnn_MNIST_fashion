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

#An accuracy and lost evolution graph is represented:
plt.title('Precisión')
plt.plot(hist_model.history['accuracy'], color='blue', label='train')
plt.plot(hist_model.history['val_accuracy'], color='orange', label='test')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.title('Loss')
plt.plot(hist_model.history['loss'], color='blue', label='train')
plt.plot(hist_model.history['val_loss'], color='orange', label='test')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Finally, we use the test set to evaluate the model
_, acc = model.evaluate (x_test, y_test, verbose = 0)
print ('%. 3f'% (acc * 100.0))
 
#We use the obtained model to predict the first four dataset images.
print(model.predict(x_test[:4]))
y_test[:4]

#Step 3: Mini-VGG model creation, compile, training and evaluation

#A convolutional neural network based on VGG will be implemented. We´ll use a secuential model divided in 4 parts:

#-First: CONV (32 kernels of size 3x3 and stride 1) => RELU => CONV => RELU => POOL (2x2, stride 2). After the activation layers we will add Batch Normalization (chanDim = -1) and after the pooling layer we will add a dropout = 0.25.
#-Second: Same as the first part, but doubling the number of "feature maps" or kernels in the convolutional layers.
#-Third: FC (Fully connected of 512 neurons) => RELU. Add Batch Normalization and Dropout = 0.5.
#-Fourth: Finally we add the classifier (softmax with the corresponding number of outputs).
def create_mini_vgg ():
    #First part of the mini-vgg
    #Sequential model is created
    model = Sequential ()
    #Since the first layer of this cnn is a 32 kernel convolution layer, it is passed as an argument the
    # input_shape parameter, indicating that the input images will be 28x28 dimension and grayscale
    model.add (Conv2D (32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    #After the first convolutional layer, we introduce a normalization layer. in this way we normalize
    #the entrance to the second convolutional layer. As a consequence, the speed in the
    #training. The chanDim parameter refers to the axis attribute of the Batch Normalization layer.
    model.add (BatchNormalization ())
    # We add the second convolutional layer
    model.add (Conv2D (32, (3, 3), activation = 'relu'))
    # Again, we normalize the input to the max pooling layer.
    model.add (BatchNormalization ())
    #A max pooling layer is added with stride = 2
    model.add (MaxPooling2D ((2, 2), strides = 2))
    #Finally to complete the first part of the cnn mini-vgg, we add a Dropout layer with probability
    # 0.25
    model.add (Dropout (0.25))
    
    #Second part of the mini-vgg
    # The first part is replicated, but in this case we double the number of kernels, that is, we went from 32 to 64.
    model.add (Conv2D (64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    # Again, we normalize the input to the second convolutional layer.
    model.add (BatchNormalization ())
    # We add the second convolutional layer with 64 kernels.
    model.add (Conv2D (64, (3, 3), activation = 'relu'))
    # Again, we normalize the input to the max pooling layer.
    model.add (BatchNormalization ())
    #A max pooling layer is added with stride = 2.
    model.add (MaxPooling2D ((2, 2), strides = 2))
    #Finally to complete the first part of the cnn mini-vgg, we add a Dropout layer with probability
    # 0.25.
    model.add (Dropout (0.25))
    
    #Third part of the mini-vgg
    #In this part the fully connected layer is created. In this layer take an input set, in this case,
    #from part 2 of the mini-vgg, and outputs an n-dimensional vector, where n is the number of classes
    #a classify.
    #Before adding the FC layer, we add a flatten layer in order to transform the two-dimensional matrix into a vector
    model.add (Flatten ())
    # FC layer with 512 neurons with batch normalization and dropout.
    model.add (Dense (512, activation = 'relu'))
    model.add (BatchNormalization ())
    model.add (Dropout (0.5))
    
    # Fourth of the mini-vgg
    #Finally, we add the output layer with 10 nodes and a softmax activation function, in order to find the
    # distribution of the probability of classifying a given image among 10 classes.
    model.add (Dense (10, activation = 'softmax'))
    
    return model
