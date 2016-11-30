'''Code for fine-tuning Inception V3 for a new task.

Start with Inception V3 network, not including last fully connected layers.

Train a simple fully connected layer on top of these.


'''

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.applications.inception_v3 import InceptionV3, decode_predictions

# TO DO:: Replace these with paths to the downloaded data.
# Training directory
train_dir = '../../NewPhotos/train'
# Testing directory
test_dir = '../../NewPhotos/validation'

N_CLASSES = 3
IMSIZE = (299, 299)

# Start with an Inception V3 model, not including the final softmax layer.
base_model = InceptionV3(weights='imagenet')
#print 'Loaded Inception model'

# Add on new fully connected layers for the output classes.
x = Dense(32, activation='relu')(base_model.get_layer('flatten').output)
x = Dropout(0.5)(x)
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)

model.load_weights('baseline.h5')

# Show some debug output
#print (model.summary())

#print 'Trainable weights'
#print model.trainable_weights

def preprocess_input(x):
    x /= 255.
    #x -= 0.5  THIS IS BAD Y IS THIS HERE
    #x *= 2.
    return x

import html
import glob
import progress
htmltext = html.getHead()
images = glob.glob(test_dir + "/*/*.jpg")

count = 0
totalcorrect = 0
progress.progress_bar(0)
for image_path in images:
    img = image.load_img(image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)[0]

    correct = 0
    if image_path.__contains__("Rock"):
        correct = 1
    elif image_path.__contains__("Scissors"):
        correct = 2

    htmltext += html.makeRow(image_path, preds, correct, np.argmax(preds))
    count += 1
    if correct == np.argmax(preds): totalcorrect += 1
    progress.progress_bar(count*100/len(images))

htmltext += html.getTail(totalcorrect,count)

outfile = open("test.html", 'w')
outfile.write(htmltext)
outfile.close()

print "Score: %d/%d" %(totalcorrect, count)

htmltext = html.getHead()
images = glob.glob(train_dir + "/*/*.jpg")

count = 0
totalcorrect = 0
progress.progress_bar(0)
for image_path in images:
    img = image.load_img(image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)[0]

    correct = 0
    if image_path.__contains__("Rock"):
        correct = 1
    elif image_path.__contains__("Scissors"):
        correct = 2

    htmltext += html.makeRow(image_path, preds, correct, np.argmax(preds))
    count += 1
    if correct == np.argmax(preds): totalcorrect += 1
    progress.progress_bar(count*100/len(images))

htmltext += html.getTail(totalcorrect,count)

outfile = open("train.html", 'w')
outfile.write(htmltext)
outfile.close()

print "Score: %d/%d" %(totalcorrect, count)




