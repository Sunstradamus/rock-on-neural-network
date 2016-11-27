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

# print 'Loading Rock class test images'
# cntr = 0
# fail = 0
# for image in sorted(os.listdir(TEST_DIR+'/Rock')):
# 	img = cv2.imread(TEST_DIR+'/Rock/'+image)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	#print 'Creating keypoints'
# 	kp, des = sift.detectAndCompute(img, None)
# 	#print 'Calculating best match'
# 	max_count = 0
# 	classifier = ''
# 	for classname in train_data:
# 		for (kp_t,des_t) in train_data[classname]:
# 			matches = matcher.knnMatch(des, des_t, k=2)

# 			count = 0
# 			for (m,n) in matches:
# 				if m.distance < 0.8*n.distance:
# 					count += 1
# 			if count > max_count:
# 				max_count = count
# 				classifier = classname
# 	if classifier != 'Rock':
# 		fail += 1
# 	cntr += 1
# 	print 'Best match ('+image+'): ' + classifier + ' with score ' + str(max_count)
# print 'Failure rate ' + str(fail) + '/' + str(cntr)

print 'Loading Paper class test images'
min_fail = 99
min_delta = 0
for delta in np.arange(0.5,1,0.01):
	cntr = 0
	fail = 0
	for image in sorted(os.listdir(TEST_DIR+'/Paper')):
		img = cv2.imread(TEST_DIR+'/Paper/'+image)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#print 'Creating keypoints'
		kp, des = sift.detectAndCompute(img, None)
		#print 'Calculating best match'
		max_count = 0
		classifier = ''
		for classname in train_data:
			for (kp_t,des_t) in train_data[classname]:
				matches = matcher.knnMatch(des, des_t, k=2)

				count = 0
				for (m,n) in matches:
					if m.distance < delta*n.distance:
						count += 1
				if count > max_count:
					max_count = count
					classifier = classname
		if classifier != 'Paper':
			fail += 1
		cntr += 1
		#print 'Best match ('+image+'): ' + classifier + ' with score ' + str(max_count)
		if fail < min_fail:
			min_fail = fail
			min_delta = delta
	print 'Failure rate ('+str(delta)+') ' + str(fail) + '/' + str(cntr)
print 'Min delta ' + str(min_delta) + ' with failures ' + str(min_fail)

# print 'Loading Scissors class test images'
# cntr = 0
# fail = 0
# for image in sorted(os.listdir(TEST_DIR+'/Scissors')):
# 	img = cv2.imread(TEST_DIR+'/Scissors/'+image)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	#print 'Creating keypoints'
# 	kp, des = sift.detectAndCompute(img, None)
# 	#print 'Calculating best match'
# 	max_count = 0
# 	classifier = ''
# 	for classname in train_data:
# 		for (kp_t,des_t) in train_data[classname]:
# 			matches = matcher.knnMatch(des, des_t, k=2)

# 			count = 0
# 			for (m,n) in matches:
# 				if m.distance < 0.8*n.distance:
# 					count += 1
# 			if count > max_count:
# 				max_count = count
# 				classifier = classname
# 	if classifier != 'Scissors':
# 		fail += 1
# 	cntr += 1
# 	print 'Best match ('+image+'): ' + classifier + ' with score ' + str(max_count)
# print 'Failure rate ' + str(fail) + '/' + str(cntr)