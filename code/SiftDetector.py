import cv2
import numpy as np
import os
import platform

class SiftDetector(object):
    """docstring for SiftDetector"""
    steps = {'Rock': 0.78, 'Paper': 0.73, 'Scissors': 0.74}
    ratio = 0.76 # 0.78 = without Mori photos, 0.76 with
    train_dir = ""
    train_data = dict()

    def __init__(self, debug=False):
        super(SiftDetector, self).__init__()
        if platform.system() == "Darwin":
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT()
        self.matcher = cv2.BFMatcher()
        self.debug = debug

    def drawMatches(self, img1, kp1, img2, kp2, matches):
        """
        @http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python/26227854#26227854

        My own implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0

        This function takes in two images with their associated 
        keypoints, as well as a list of DMatch data structure (matches) 
        that contains which keypoints matched in which images.

        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.

        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.

        img1,img2 - Grayscale images
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
        """

        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

        # Place the first image to the left
        out[:rows1,:cols1] = np.dstack([img1, img1, img1])

        # Place the next image to the right of it
        out[:rows2,cols1:] = np.dstack([img2, img2, img2])

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


        # Show the image
        cv2.imshow('Matched Features', out)
        cv2.waitKey(0)
        cv2.destroyWindow('Matched Features')

        # Also return the image if you'd like a copy
        return out

    def loadTrainingImages(self, TRAIN_DIR):
        """
        dir - String of relative path to folder that contains subfolders of training classes
        """
        self.train_dir = TRAIN_DIR
        for classname in sorted(os.listdir(TRAIN_DIR)):
            if classname == ".DS_Store": continue
            self.train_data[classname] = list()
            for image in sorted(os.listdir(TRAIN_DIR+'/'+classname)):
                if image == ".DS_Store": continue
                img = cv2.imread(TRAIN_DIR+'/'+classname+'/'+image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.train_data[classname].append(self.sift.detectAndCompute(img, None))
        if self.debug:
            print 'Loaded ' + str(len(self.train_data)) + ' classes'

    def predict(self, img, rps=False):
        """
        img - color image array
        """
        if len(self.train_data) == 0:
            raise RuntimeError('SiftDetector has not learned any classifiers yet, call loadTrainingData before predict')
        if rps:
            if self.debug:
                print 'Using RPS predict'
            return self.__rpsPredict(img)
        else:
            if self.debug:
                print 'Using generic predict'
            return self.__genericPredict(img)

    def predictAndDraw(self, img, rps=False):
        """
        img - color image array
        """
        if len(self.train_data) == 0:
            raise RuntimeError('SiftDetector has not learned any classifiers yet, call loadTrainingData before predict')
        if rps:
            if self.debug:
                print 'Using RPS predict'
            (cls, kp, index, kp_t, matches) = self.__rpsPredict(img, True)
            good = list()
            for (m,n) in matches:
                if m.distance < self.steps[cls]*n.distance:
                    good.append(m)
        else:
            if self.debug:
                print 'Using generic predict'
            (cls, kp, index, kp_t, matches) = self.__genericPredict(img, True)
            good = list()
            for (m,n) in matches:
                if m.distance < self.ratio*n.distance:
                    good.append(m)

        images = sorted(os.listdir(self.train_dir+'/'+cls))
        if images.__contains__(".DS_Store"): images.remove(".DS_Store")
        image = cv2.imread(self.train_dir+'/'+cls+'/'+images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.drawMatches(gray, kp, image, kp_t, good)
        return cls


    def calcGlobalRatio(self, data_dir, start_step=0.5, step_size=0.01):
        if len(self.train_data) == 0:
            raise RuntimeError('SiftDetector has not learned any classifiers yet, call loadTrainingData before calcGlobalRatio')

        best_delta = 0
        best_fail = 999
        test_data = self.__loadData(data_dir, True)
        for delta in np.arange(start_step, 1, step_size):
            fail = 0

            for class_name in test_data:
                for (kp, des) in test_data[class_name]:
                    max_count = 0
                    classifier = ''
                    for classname in self.train_data:
                      for (kp_t,des_t) in self.train_data[classname]:
                          matches = self.matcher.knnMatch(des, des_t, k=2)

                          count = 0
                          for (m,n) in matches:
                              if m.distance < delta*n.distance:
                                  count += 1
                          if count > max_count:
                              max_count = count
                              classifier = classname
                    if classifier != class_name:
                        fail += 1

            if fail <= best_fail:
                best_fail = fail
                best_delta = delta
            if self.debug:
                print 'Failure rate ('+str(delta)+') ' + str(fail)
        return best_delta

    def calcLocalRatio(self, data_dir, class_name, start_step=0.5, step_size=0.01):
        if len(self.train_data) == 0:
            raise RuntimeError('SiftDetector has not learned any classifiers yet, call loadTrainingData before calcLocalRatio')

        best_delta = 0
        best_fail = 999
        test_data = self.__loadData(data_dir+'/'+class_name)
        for delta in np.arange(start_step, 1, step_size):
            fail = 0

            for (kp, des) in test_data:
                max_count = 0
                classifier = ''
                for classname in self.train_data:
                  for (kp_t,des_t) in self.train_data[classname]:
                      matches = self.matcher.knnMatch(des, des_t, k=2)

                      count = 0
                      for (m,n) in matches:
                          if m.distance < delta*n.distance:
                              count += 1
                      if count > max_count:
                          max_count = count
                          classifier = classname
                if classifier != class_name:
                    fail += 1

            if fail <= best_fail:
                best_fail = fail
                best_delta = delta
            if self.debug:
                print 'Failure rate ('+str(delta)+') ' + str(fail)
        return best_delta

    def __genericPredict(self, img, return_metadata=False):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(image, None)
        image_idx = 0
        image_matches = None
        matched_kp = None
        max_count = 0
        classifier = ''
        for classname in self.train_data:
          img_num = 0
          for (kp_t,des_t) in self.train_data[classname]:
              matches = self.matcher.knnMatch(des, des_t, k=2)

              count = 0
              for (m,n) in matches:
                  if m.distance < self.ratio*n.distance:
                      count += 1
              if count > max_count:
                  max_count = count
                  classifier = classname
                  if return_metadata:
                    image_idx = img_num
                    image_matches = matches
                    matched_kp = kp_t
              img_num += 1
        if return_metadata:
            return classifier, kp, image_idx, matched_kp, image_matches
        else:
            return classifier

    def __rpsPredict(self, img, return_metadata=False):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(image, None)
        image_idx = 0
        image_matches = None
        matched_kp = None
        max_count = 0
        classifier = ''
        for classname in self.train_data:
          img_num = 0
          for (kp_t,des_t) in self.train_data[classname]:
              matches = self.matcher.knnMatch(des, des_t, k=2)

              count = 0
              for (m,n) in matches:
                  if m.distance < self.steps[classname]*n.distance:
                      count += 1
              if count > max_count:
                  max_count = count
                  classifier = classname
                  if return_metadata:
                    image_idx = img_num
                    image_matches = matches
                    matched_kp = kp_t
              img_num += 1
        if return_metadata:
            return classifier, kp, image_idx, matched_kp, image_matches
        else:
            return classifier

    def __loadData(self, ddir, globald=False):
        if globald:
            data = dict()
            for classname in os.listdir(ddir):
                if classname == ".DS_Store": continue
                data[classname] = list()
                for image in os.listdir(ddir+'/'+classname):
                    if image == ".DS_Store": continue
                    img = cv2.imread(ddir+'/'+classname+'/'+image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    data[classname].append(self.sift.detectAndCompute(img, None))
        else:
            data = list()
            for image in os.listdir(ddir):
                if image == ".DS_Store": continue
                img = cv2.imread(ddir+'/'+image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                data.append(self.sift.detectAndCompute(img, None))
        return data
