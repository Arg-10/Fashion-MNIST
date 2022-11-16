#!/usr/bin/env python
# coding: utf-8

# ## Importing the relevant packages

# In[1]:


import tensorflow as tf

# TensorFLow has a data provider for Fasion_MNIST that we'll use.
from tensorflow.keras.datasets import fashion_mnist

import tensorflow_datasets as tfds
fmnist_dataset, fmnist_info = tfds.load(name='fashion_mnist', with_info=True, as_supervised=True)


# ## Data

# In[2]:


# with_info=True will also provide us with a tuple containing information about the version, features, number of samples
# as_supervised=True will load the dataset in a 2-tuple structure (input, target) 

fmnist_train, fmnist_test = fmnist_dataset['train'], fmnist_dataset['test']

# by default, TF has training and testing datasets, but no validation sets; thus we must split it on our own.

num_validation_samples = 0.1 * fmnist_info.splits['train'].num_examples
# let's cast this number to an integer, as a float may cause an error along the way
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = fmnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


# We would like to scale our data to make the result more numerically stable, here, we'll simply prefer to have inputs between 0 and 1
def scale(image, label):
    image = tf.cast(image, tf.float32)
    # since the possible values for the inputs are 0 to 255 (256 different shades of grey), by dividing each element by 255, all elements will be between 0 and 1 
    image /= 255.

    return image, label


# We would map a the transformation
scaled_train_and_validation_data = fmnist_train.map(scale)

# Finally, we scale and batch the test data
test_data = fmnist_test.map(scale)


# let's also shuffle the data

BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

# We take our validation data to be equal to 10% of the training set
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)

train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

# We set the batch size
BATCH_SIZE = 150

train_data = train_data.batch(BATCH_SIZE)

validation_data = validation_data.batch(num_validation_samples)

# Then, we batch the test data
test_data = test_data.batch(num_test_samples)

# Now, we take the next batch, because for as_supervized=True, we've got a 2-tuple structure
validation_inputs, validation_targets = next(iter(validation_data))


# ## Model

# ###  Outlining the model

# In[3]:


input_size = 784
output_size = 10
hidden_layer_size1 = 512
hidden_layer_size2 = 512
    
# Defining how the model will look like
# 'Flatten' simply takes our 28x28x1 tensor and orders it into a (None,) or (28x28x1,) = (784,) vector; this allows us to actually create a feed forward neural network
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input layer
    tf.keras.layers.Dense(hidden_layer_size1, activation='tanh'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size2, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(output_size, activation='softmax')]) # output layer


# ### Choosing the optimizer and the loss function

# In[4]:


# We define the optimizer, the loss function, and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# ### Training

# In[5]:


# Enter the maximum number of epochs
NUM_EPOCHS = 10

# we fit the model
model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose =2)


# ## Testing the model

# In[6]:


test_loss, test_accuracy = model.evaluate(test_data)


# In[7]:


# We print the test loss and accuracy
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


# ### Finding the best hyperparameters

# In[ ]:


from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

def model_builder (hp):          #hp means hyper parameters
    model=Sequential()
    model.add(Flatten(input_shape=(28,28)))
    #providing range for number of neurons in a hidden layer
    model.add(Dense(units=hp.Int('num_of_neurons',min_value=32,max_value=512,step=32),
                                    activation='relu'))
    #output layer
    model.add(Dense(10,activation='softmax'))
    #compiling the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

#installing the required libraries
from tensorflow import keras
from keras_tuner import Hyperband

tuner = Hyperband(model_builder,
objective='val_accuracy',
max_epochs=10,
factor=3,
directory='keras_tuner',
project_name='fashion_mnist',
overwrite=True)

#this tells us how many hyperparameter we are tuning
#in our case it's 2 = neurons,learning rate
tuner.search_space_summary()

#fitting the tuner on train dataset
tuner.search(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

tuner.results_summary()


# In[ ]:




