import numpy as np
import cv2
from NN import imagenet_finetune

# Turn on the camera
cap = cv2.VideoCapture(0)

# take first frame of the video
track_window = (0, 0, 299, 299)

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

NN_on = False

while(1):
    ret ,frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0, 1],hist,[0,hmax, 0, smax],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    if NN_on:

        # get image from track_window and convert to the right format
        (x,y,w,h) = track_window
        part = frame[y:y+h,x:x+w,:].astype("float32")
        x = np.expand_dims(part, axis=0)
        # Pass it through the neural network
        x = imagenet_finetune.inception.preprocess_input(x)
        preds = imagenet_finetune.model.predict(x)
        # Output the results
        print('Predicted:', preds)

    # Draw the rectangle on the image
    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv2.imshow('img2',img2)

    # Read keyboard input (q=quit, n=toggle neural network on/off default=off)
    k = cv2.waitKey(60) & 0xff
    if k == ord('q'):
        break
    if k == ord('n'):
        NN_on = not NN_on

# Turn the camera off and close the window
cv2.destroyAllWindows()
cap.release()
