import numpy as np
import glob
import cv2
import classify
import SiftDetector

# Initialize the SIFT detector
print "Initializing SIFT..."
#TRAIN_DIR = 'NewPhotos/train'
TRAIN_DIR = 'Models'
s = SiftDetector.SiftDetector()
s.loadTrainingImages(TRAIN_DIR)
#s.calcGlobalRatio("NewPhotos/validation")

# Turn on the camera
cap = cv2.VideoCapture(0)

track_window = (400, 100, 299, 299)

# Load the image
(hmax, smax) = (255, 255)
(hbin, sbin) = (180, 4)
hists = []
for image in glob.glob("./Models/Rock/1.jpg"):
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

NN_on = False
SIFT_on = False
font = cv2.FONT_HERSHEY_SIMPLEX
text1 = ""
text2 = ""

while(1):
    ret ,frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = np.zeros(hsv.shape[0:2])
    for hist in hists:
        dst += cv2.calcBackProject([hsv],[0,1],hist,[0,hmax, 0, smax],1)/(100.0*len(hists))

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    if NN_on:

        # get image from track_window and convert to the right format
        (x,y,w,h) = track_window
        part = frame[y:y+h,x:x+w,:].astype("float32")
        x = np.expand_dims(part, axis=0)
        # Pass it through the neural network
        x = classify.preprocess_input(x)
        preds = classify.model.predict(x)
        # Output the results
        options = ["Paper", "Rock", "Scissors"]
        text1 = options[np.argmax(preds)]
        print('Predicted:', preds)
        NN_on = False
    if SIFT_on:

        # get image from track_window and convert to the right format
        (x,y,w,h) = track_window
        part = frame[y:y+h,x:x+w,:]
        # Pass it through SIFT
        classifier = s.predict(part)
        # Output the results
        #print('Predicted:', classifier)
        text2 = classifier
        #SIFT_on = False

    # Draw the rectangle on the image
    cv2.putText(frame,text1,(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,text2,(10,300), font, 4,(255,0,255),2,cv2.LINE_AA)
    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv2.imshow('img2',img2)

    # Read keyboard input (q=quit, n=toggle neural network on/off default=off)
    k = cv2.waitKey(60) & 0xff
    if k == ord('q'):
        break
    if k == ord('n'):
        NN_on = not NN_on
    if k == ord('s'):
        SIFT_on = not SIFT_on

# Turn the camera off and close the window
cv2.destroyAllWindows()
cap.release()
