import numpy as np
import cv2
import glob
import os

# Load the image
frame = cv2.imread("./Hand.jpg")
# Convert to HSV colorspace
hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Specify bin size (we don't use the V component since it is not invariant to light
(hmax, smax) = (180, 200)
(hbin, sbin) = (180, 4)
# mask out all the dark pixels, only use pixel values above (0, 60, 60)
mask = cv2.inRange(hsv, np.array((0., 60.,60.)), np.array((hmax,smax,255.)))
# create a histogram over just the hue values binning them into 255 bins
hist = cv2.calcHist([hsv],[0,1],mask,[hbin, sbin],[0,hmax, 0, smax])
# Normalize the histogram to be between 0 and 255
cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# load the images
images = glob.glob("./Photos/*/*.jpg")

# If you want to look at each image individually and rate them as correct or not
manual_check = False
count = 0
correct = 0

for filename in images:
    # Get the current frame and convert to hsv colorspace
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Backpropagate - returns a probability density image
    dst = cv2.calcBackProject([hsv],[0,1],hist,[0,hmax, 0, smax],1)

    # apply meanshift to find the new location (using a fixed start location)
    ret, track_window = cv2.meanShift(dst, (400,100,299,299), term_crit)
    x,y,w,h = track_window

    if manual_check:
        # Draw the rectangle on the image
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
    
        # Wait until y or n is pressed (y=hand is in square, n=hand is not in square)
        count += 1
        while True:
            k=cv2.waitKey(1)
            if k == ord('y'):
                correct += 1
                break
            elif k == ord('n'):
                print filename
                break

    # Save the cropped image
    if ((not manual_check) or k == 'y'):
        cv2.imwrite(filename.replace("Photos", "NewPhotos"),frame[y:y+h,x:x+w,:])

# Print the accuracy
if manual_check:
    print correct*100/count, "% correct"
