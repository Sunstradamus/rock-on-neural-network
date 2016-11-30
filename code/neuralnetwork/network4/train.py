'''Code for fine-tuning Inception V3 for a new task.'''

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.optimizers import SGD


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

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Show some debug output
#print (model.summary())

#print 'Trainable weights'
#print model.trainable_weights

# Data generators for feeding training/testing images to the model.
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        #target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=60,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,  # this is the target directory
        #target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=60,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=215,
        nb_epoch=50,
        validation_data=test_generator,
        verbose=2,
        nb_val_samples=196)


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators for feeding training/testing images to the model.
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        #target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=60,
	classes=['Paper', 'Rock', 'Scissors'],
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,  # this is the target directory
        #target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=60,
	classes=['Paper', 'Rock', 'Scissors'],
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=215,
        nb_epoch=300,
        validation_data=test_generator,
        verbose=2,
        nb_val_samples=196)

model.save_weights('network4.h5')  # always save your weights after training or during training


























