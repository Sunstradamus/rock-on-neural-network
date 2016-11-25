import cv2
import numpy as np

img = cv2.imread('Hand.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(img_gray, None)

img = cv2.drawKeypoints(img_gray, kp)
cv2.imshow('kp', img)
cv2.waitKey(0)

# kp, des = sift.detectAndCompute(img_gray, None)