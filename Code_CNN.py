#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
#import time
#import tensorflow as tf
#import argparse
#from tensorflow.keras.callbacks import TensorBoard
import numpy as np
#import random
#from tensorflow.keras.preprocessing.image import img_to_array, load_img
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from tensorflow.keras import layers
#from tensorflow.keras import Model


# In[2]:


#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs\{}', histogram_freq=1)-

#gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
#sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# Using CPU only- 
#with tf.device("/cpu:0"):-
#start_time = time.time()
#img_dim = 150
#tBS = 20
#vBS = 20
#tSS = 6400/tBS
#vSS = 1600/tSS


# In[3]:


#Name = "food and drink classification-{}".format(int(time.time()))

#tensorboard = TensorBoard(log_dir='{}'.format(Name))


# In[3]:


#base_dir = 'C:\Yelp Images'
#train_dir = 'C:\Yelp Images\Train'
#validation_dir = 'C:\Yelp Images\Test'

train_food_dir = 'C:\Yelp Images\Train\Food_train'

#train_drink_dir = 'C:\Yelp Images\Train\Drink_train'

#validation_food_dir = 'C:\Yelp Images\Test\Food_test'

#validation_drink_dir = 'C:\Yelp Images\Test\Drink_test'

train_food_fnames = os.listdir(train_food_dir)

#train_drink_fnames = os.listdir(train_drink_dir)
#train_drink_fnames.sort()


# In[5]:



pic_index = 0
img_dim=150
img_input = layers.Input(shape=(img_dim, img_dim, 3))


# In[ ]:


for img in os.listdir(train_food_dir):
    img_array=cv2.imread(os.path.join(train_food_dir, img))
    plt.imshow(img_array)
    plt.show()


# In[ ]:


# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Dropout(0.33)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Dropout(0.33)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(64, activation='relu')(x)

x = layers.Dropout(0.4)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)


# In[7]:


# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
model = Model(img_input, output)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[8]:


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


# In[9]:


history = model.fit_generator(
      train_generator,
#      steps_per_epoch=100,  # 2000 images = batch_size * steps
      steps_per_epoch=tSS,  # 800 images = batch_size * steps
#      epochs=15,
      epochs=15,
      validation_data=validation_generator,
      validation_steps=vSS,
      callbacks = [tensorboard],  # 400 images = batch_size * steps
      verbose=2)

print("\n Total execution time:  %s s ---" % (time.time() - start_time))
print('Training Size = ',tBS,', Testing Size = ',vBS)


# In[80]:


# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

#        epochs = range(len(acc))
#        #plt.figure()
#        
#        #Plot training and validation accuracy per epoch
#        plt.plot(epochs, acc)
#        plt.plot(epochs, val_acc)
#        plt.title('Training and validation accuracy')
#        plt.savefig('Accuracy.png')
#        
#        #plt.figure()
#        
#        # Plot training and validation loss per epoch
#        plt.plot(epochs, loss)
#        plt.plot(epochs, val_loss)
#        plt.title('Training and validation loss')
#        plt.savefig('Loss.png')


# In[ ]:





# In[ ]:




