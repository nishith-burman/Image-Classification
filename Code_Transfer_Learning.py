#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from tensorflow.keras import layers
from tensorflow.keras import Model


# In[2]:


from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'C:\\Users\\nishi\\wgetdown\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)


# In[3]:


for layer in pre_trained_model.layers:
  layer.trainable = False


# In[4]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# In[16]:


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(128, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])


# In[17]:


import cv2
import os
import time
import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model


gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# Using CPU only 
#with tf.device("/cpu:0"):
start_time = time.time()
img_dim = 150
tBS = 20
vBS = 20
tSS = 6400/tBS
vSS = 1600/tSS

Name = "cd-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logab{}'.format(Name))

base_dir = 'C:\Yelp Images'
train_dir = 'C:\Yelp Images\Train'
validation_dir = 'C:\Yelp Images\Test'

train_food_dir = 'C:\Yelp Images\Train\Food_train'

train_drink_dir = 'C:\Yelp Images\Train\Drink_train'

validation_food_dir = 'C:\Yelp Images\Test\Food_test'

validation_drink_dir = 'C:\Yelp Images\Test\Drink_test'

train_food_fnames = os.listdir(train_food_dir)

train_drink_fnames = os.listdir(train_drink_dir)
train_drink_fnames.sort()


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(img_dim, img_dim),  # All images will be resized to 150x150
        batch_size=tBS,
#        batch_size=8,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_dim, img_dim),
#        batch_size=20,
        batch_size=vBS,
        class_mode='binary')


# In[18]:


model.summary()


# In[ ]:


history = model.fit_generator(
      train_generator,
#      steps_per_epoch=100,  # 2000 images = batch_size * steps
      steps_per_epoch=tSS,  # 800 images = batch_size * steps
#      epochs=15,
      epochs=5,
      validation_data=validation_generator,
      validation_steps=vSS,
      callbacks = [tensorboard],  # 400 images = batch_size * steps
      verbose=2)


# In[ ]:




