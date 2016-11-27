import cv2
import numpy as np
import os

TRAIN_DIR = 'NewPhotos/train'
TEST_DIR = 'NewPhotos/validation'
TRAIN_FILES = dict()
TEST_FILES = dict()

print 'Loading training data'
for classname in sorted(os.listdir(TRAIN_DIR)):
	TRAIN_FILES[classname] = list()

	for image in os.listdir(TRAIN_DIR+'/'+classname):
		img = cv2.imread(TRAIN_DIR+'/'+classname+'/'+image)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		TRAIN_FILES[classname].append(img)

	print 'Processed training class "' + classname + '"'
print

print 'Initializing SIFT keypoints'
sift = cv2.SIFT()
matcher = cv2.BFMatcher()
train_data = dict()
for classname in TRAIN_FILES:
	train_data[classname] = list()

	for image in TRAIN_FILES[classname]:
		train_data[classname].append(sift.detectAndCompute(image, None))

	print 'Created keypoints for class "' + classname + '"'
print

print 'Loading random test image'
img = cv2.imread(TEST_DIR+'/Rock/P_111.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print 'Creating keypoints'
kp, des = sift.detectAndCompute(img, None)
print 'Calculating best match'
max_count = 0
classifier = ''
for classname in train_data:
	for (kp_t,des_t) in train_data[classname]:
		matches = matcher.knnMatch(des, des_t, k=2)

		count = 0
		for (m,n) in matches:
			if m.distance < 0.8*n.distance:
				count += 1
		if count > max_count:
			max_count = count
			classifier = classname
print 'Best match: ' + classifier + ' with score ' + str(max_count)