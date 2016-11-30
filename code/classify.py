'''Code for fine-tuning Inception V3 for a new task.'''

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

# TO DO:: Replace these with paths to the downloaded data.
# Training directory
train_dir = '/media/wilson/5074-42A6/419porj/419-project/code/NewPhotos/train'
# Testing directory
test_dir = '/media/wilson/5074-42A6/419porj/419-project/code/NewPhotos/validation'

N_CLASSES = 3
IMSIZE = (299, 299)


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
# let's add a fully-connected layer
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 3 classes
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

model.load_weights('network1.h5')

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x




























