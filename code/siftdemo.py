import cv2
import os
import SiftDetector

TRAIN_DIR = 'NewPhotos/train'
TEST_DIR = 'NewPhotos/validation'

TEST_FILES = dict()

s = SiftDetector.SiftDetector()
print 'Started SiftDetector'
s.loadTrainingImages(TRAIN_DIR)

print 'Loading test data'
for classname in sorted(os.listdir(TEST_DIR)):
	TEST_FILES[classname] = list()

	for image in os.listdir(TEST_DIR+'/'+classname):
		img = cv2.imread(TEST_DIR+'/'+classname+'/'+image)
		TEST_FILES[classname].append(img)

	print 'Processed test class "' + classname + '"'
print

results = list()
for classname in TEST_FILES:
	failed = 0
	total = 0
	for img in TEST_FILES[classname]:
		classifier = s.predict(img)
		if classifier != classname:
			failed += 1
			print 'Incorrectly identified ' + classname + ' as ' + classifier
		else:
			print 'Correctly identified ' + classname
		total += 1
	results.append('Failure rate of "' + classname + '" is ' + str(failed) + '/' + str(total))
print
for result in results:
	print result

img = cv2.imread(TEST_DIR+'/Paper/45.jpg')
s.predictAndDraw(img)