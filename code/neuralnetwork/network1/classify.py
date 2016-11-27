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
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(32, activation='relu')(x)
x = Dropout(0.8)(x)
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

import html
import glob
import progress
htmltext = html.getHead()
images = glob.glob(train_dir + "/*/*.jpg")

count = 0
progress.progress_bar(0)
for image_path in images:
    img = image.load_img(image_path, target_size=IMSIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print preds
    print np_utils.probas_to_classes(preds)
    preds = preds[0]

    correct = 0
    if image_path.__contains__("Rock"):
        correct = 1
    elif image_path.__contains__("Scissors"):
        correct = 2

    htmltext += html.makeRow(image_path, preds, correct, np.argmax(preds))
    count += 1
    progress.progress_bar(count*100/len(images))

htmltext += html.getTail()

outfile = open("test.html", 'w')
outfile.write(htmltext)
outfile.close()




























