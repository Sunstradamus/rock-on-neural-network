import numpy as np
import cv2
import glob
import os

# Load the image
(hmax, smax) = (255, 255)
(hbin, sbin) = (180, 4)
hists = []
for image in glob.glob("./Models/*/*.jpg"):
    frame = cv2.imread(image)
    # Convert to HSV colorspace
    hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Specify bin size (we don't use the V component since it is not invariant to light
    # mask out all the dark pixels, only use pixel values above (0, 60, 60)
    mask = cv2.inRange(hsv, np.array((0., 10.,10.)), np.array((hmax,smax,255.)))
    # create a histogram over just the hue values binning them into 255 bins
    hist = cv2.calcHist([hsv],[0,1],mask,[hbin, sbin],[0,hmax, 0, smax])
    # Normalize the histogram to be between 0 and 255
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    hists.append(hist)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# load the images
images = glob.glob("./Photos/*/*.jpg")

# If you want to look at each image individually and rate them as correct or not
manual_check = True
count = 0
correct = 0

for filename in images:
    # Get the current frame and convert to hsv colorspace
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Backpropagate - returns a probability density image
    dst = np.zeros(hsv.shape[0:2])
    for hist in hists:
        dst += cv2.calcBackProject([hsv],[0,1],hist,[0,hmax, 0, smax],1)/(100.0*len(hists))

    # apply meanshift to find the new location (using a fixed start location)
    ret, track_window = cv2.meanShift(dst, (400,100,299,299), term_crit)
    x,y,w,h = track_window
    #dst = cv2.boxFilter(dst, -1, (299, 299))
    #(y, x) = np.unravel_index(dst.argmax(), dst.shape)

    if manual_check:
        # Draw the rectangle on the imagen
        #img2 = cv2.rectangle(frame, (x-150,y-150), (x+149,y+149), (0,255,0),2)
        img2 = cv2.rectangle(frame, (x-2,y-2), (x+w+1,y+h+1), (0,255,0),2)
        #cv2.imshow('img2',dst)
        #cv2.waitKey(0)
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
    if ((not manual_check) or k == ord('y')):
        cv2.imwrite(filename.replace("Photos", "NewPhotos"),frame[y:y+h,x:x+w,:])

# Print the accuracy
if manual_check:
    print correct*100/count, "% correct"
