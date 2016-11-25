import random
import glob
import os

# take 3/4 of the images from from each category of NewImage/validation and move them to NewImage/train
def movetotrain():
    for t in ["Paper", "Rock", "Scissors"]:
        images = glob.glob("./NewPhotos/validation/%s/*.jpg" %(t))
        k = random.sample(images, (len(images)*3)/4)
        for image in k:
            os.rename(image, image.replace("validation", "train"))

# move all the /train images back to /validation
def moveback():
    images = glob.glob("./NewPhotos/train/*/*.jpg")
    for image in images:
        os.rename(image, image.replace("train", "validation"))
                           
movetotrain()
#moveback()
