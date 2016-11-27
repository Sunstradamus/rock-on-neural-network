'''Code for fine-tuning Inception V3 for a new task.

Start with Inception V3 network, not including last fully connected layers.

Train a simple fully connected layer on top of these.


'''

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.applications.inception_v3 import InceptionV3

# TO DO:: Replace these with paths to the downloaded data.
# Training directory
train_dir = '/media/wilson/5074-42A6/419porj/419-project/code/NewPhotos/train'
# Testing directory
test_dir = '/media/wilson/5074-42A6/419porj/419-project/code/NewPhotos/validation'

N_CLASSES = 3
IMSIZE = (299, 299)

# Start with an Inception V3 model, not including the final softmax layer.
base_model = InceptionV3(weights='imagenet')
#print 'Loaded Inception model'

# Turn off training on base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add on new fully connected layers for the output classes.
x = Dense(32, activation='relu')(base_model.get_layer('flatten').output)
x = Dropout(0.5)(x)
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# Show some debug output
#print (model.summary())

#print 'Trainable weights'
#print model.trainable_weights


# Data generators for feeding training/testing images to the model.
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=64,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,  # this is the target directory
        target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=64,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=215, # All of the samples
        nb_epoch=100,
        validation_data=test_generator,
        verbose=2,
        nb_val_samples=196) # All of the validation images

model.save_weights('baseline.h5')  # always save your weights after training or during training

'''
Epoch 90/100
3s - loss: 0.1963 - acc: 0.9577 - val_loss: 0.4627 - val_acc: 0.8173
Epoch 91/100
3s - loss: 0.2488 - acc: 0.9218 - val_loss: 0.5323 - val_acc: 0.8077
Epoch 92/100
3s - loss: 0.1980 - acc: 0.9479 - val_loss: 0.3989 - val_acc: 0.8558
Epoch 93/100
3s - loss: 0.1685 - acc: 0.9674 - val_loss: 0.5071 - val_acc: 0.8462
Epoch 94/100
3s - loss: 0.1983 - acc: 0.9544 - val_loss: 0.5049 - val_acc: 0.7981
Epoch 95/100
3s - loss: 0.1820 - acc: 0.9577 - val_loss: 0.4921 - val_acc: 0.8269
Epoch 96/100
3s - loss: 0.2106 - acc: 0.9446 - val_loss: 0.4907 - val_acc: 0.8269
Epoch 97/100
3s - loss: 0.2260 - acc: 0.9251 - val_loss: 0.5182 - val_acc: 0.8173
Epoch 98/100
3s - loss: 0.2098 - acc: 0.9577 - val_loss: 0.4420 - val_acc: 0.8558
Epoch 99/100
3s - loss: 0.1838 - acc: 0.9674 - val_loss: 0.4371 - val_acc: 0.8462
Epoch 100/100
3s - loss: 0.1915 - acc: 0.9446 - val_loss: 0.5038 - val_acc: 0.7692
'''

















