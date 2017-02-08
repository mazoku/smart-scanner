import cv2
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import skimage.segmentation as skiseg
import skimage.morphology as skimor
import numpy as np
import mahotas
import matplotlib.pyplot as plt
import hog
import helpers
import imutils
import os
import glob


class MarkerRecognizer():

    def __init__(self):
        self.model = None
        self.descriptor = hog.HOG(orientations=18, pixelsPerCell=(10, 10), cellsPerBlock=(1, 1), normalize=True)

    # def train_opencv(self, data_path='../data/models/letter-recognition.data', classifier='knn'):
    #     # Load the data, converters convert the letter to a number
    #     data = np.loadtxt(data_path, dtype='float32', delimiter=',',
    #                       converters={0: lambda ch: ord(ch) - ord('A')})
    #
    #     # split the data to two, 10000 each for train and test
    #     train, test = np.vsplit(data, 2)
    #
    #     # split trainData and testData to features and responses
    #     # responses, trainData = np.hsplit(train, [1])
    #     # labels, testData = np.hsplit(test, [1])
    #     trainLabels, trainData = np.hsplit(train, [1])
    #
    #     # take the MNIST data and construct the training and testing split, using 75% of the
    #     # data for training and 25% for testing
    #
    #     (trainData, testData, trainLabels, testLabels) = train_test_split(np.hsplit(data, [1]),
    #               np.hsplit(train, [2]), test_size=0.25, random_state=42)
    #
    #     if classifier == 'knn':
    #         # Initiate the kNN, classify, measure accuracy.
    #         # classif = cv2.KNearest()
    #         # classif.train(trainData, trainLabels)
    #         self.model = KNeighborsClassifier(n_neighbors=5)
    #         self.model.fit(trainData, trainLabels)
    #     elif classifier == 'svm':
    #         self.model = SVC(kernel="linear")
    #         self.model.fit(trainData, trainLabels)

    def deskew(self, image, width):
        # grab the width and height of the image and compute moments for the image
        (h, w) = image.shape[:2]
        moments = cv2.moments(image)

        # deskew the image by applying an affine transformation
        skew = moments["mu11"] / moments["mu02"]

        M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        # resize the image to have a constant width
        image = imutils.resize(image, width=width)

        # return the deskewed image
        return image

    def center_extent(self, image, size):
        # grab the extent width and height
        (eW, eH) = size

        # handle when the width is greater than the height
        if image.shape[1] > image.shape[0]:
            image = imutils.resize(image, width=eW)

        # otherwise, the height is greater than the width
        else:
            image = imutils.resize(image, height=eH)

        # allocate memory for the extent of the image and grab it
        extent = np.zeros((eH, eW), dtype="uint8")
        offsetX = (eW - image.shape[1]) // 2
        offsetY = (eH - image.shape[0]) // 2
        extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

        # compute the center of mass of the image and then move the center of mass to the center
        # of the image
        (cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
        (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
        M = np.float32([[1, 0, dX], [0, 1, dY]])
        extent = cv2.warpAffine(extent, M, size)

        # return the extent of the image
        return extent

    def train_user(self, markers='all', sheets_dir='../data/sheets/'):
        markers = markers.upper()
        fnames = glob.glob(sheets_dir + '*')

        # get the valid markers defined by user in the 'markers' parameter
        sheets = []
        for f in fnames:
            letter = f.split('/')[-1].split('.')[0]
            if (markers == 'ALL' and 'test' not in letter) or letter in markers:
                sheets.append((letter, cv2.imread(f, 0)))

        markers = self.sheets2data(sheets)
        labels = [x[0] for x in markers]
        data = np.array([x[1] for x in markers])

        # describe
        hogs = self.describe(data)

        # train model
        self.model = LinearSVC(random_state=42)
        self.model.fit(hogs, labels)

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

    def describe(self, data):
        features = []
        for d in data:
            hist = self.descriptor.describe(d)
            features.append(hist)
        return features

    def detect_sheet(self, img):
        im_s = cv2.bilateralFilter(img, 11, 15, 15)
        t, im_t = cv2.threshold(im_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im_closed = cv2.morphologyEx(im_t, cv2.MORPH_CLOSE, np.ones((3, 3)))

        # find and approximate the biggest contour
        cnts = cv2.findContours(im_closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # get the top-down view and threshold the markers
        warped = helpers.four_point_transform(im_s, cnt.reshape(4, 2))

        return warped

    def detect_markers(self, img):
        t, img_t = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # close to merge possibly disconnected parts of markers (just for finding contours)
        # warped_t = cv2.morphologyEx(warped_t, cv2.MORPH_CLOSE, np.ones((9, 9)))
        img_m = cv2.dilate(img_t, np.ones((5, 5)), iterations=3)
        img_m = cv2.erode(img_m, np.ones((5, 5)), iterations=3)

        # remove objects touching the borders
        img_c = skiseg.clear_border(img_m)
        img_filled = skimor.remove_small_holes(img_c, min_size=20000).astype(np.uint8)

        # plt.figure()
        # plt.subplot(121), plt.imshow(img_t, 'gray', interpolation='nearest')
        # plt.subplot(122), plt.imshow(img_filled, 'gray', interpolation='nearest')
        # plt.show()

        # find individual letters and crop them from image
        markers = []
        cnts = cv2.findContours(img_filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for c in cnts:
            if cv2.contourArea(c) < 200:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            roi = img_t[y:y + h, x:x + w]
            marker = self.deskew(roi, 20)
            marker = self.center_extent(marker, (20, 20))
            markers.append((marker, (x, y, w, h)))

        return markers

    def sheets2data(self, sheets):
        markers = []
        for label, sheet in sheets:
            # threshold the image and apply morphological operations
            # im_s = cv2.bilateralFilter(sheet, 11, 15, 15)
            # t, im_t = cv2.threshold(im_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # im_closed = cv2.morphologyEx(im_t, cv2.MORPH_CLOSE, np.ones((3, 3)))
            #
            # # find and approximate the biggest contour
            # cnts = cv2.findContours(im_closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            # cnt = sorted(cnts, key=cv2.contourArea)[-1]
            # cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            #
            # # get the top-down view and threshold the markers
            # warped = helpers.four_point_transform(im_s, cnt.reshape(4, 2))
            # t, warped_t = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # # close to merge possibly disconnected parts of markers (just for finding contours)
            # # warped_t = cv2.morphologyEx(warped_t, cv2.MORPH_CLOSE, np.ones((9, 9)))
            # warped_m = cv2.dilate(warped_t, np.ones((5, 5)), iterations=3)
            # warped_m = cv2.erode(warped_m, np.ones((5, 5)), iterations=3)

            warped = self.detect_sheet(sheet)

            # t, warped_t = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # # close to merge possibly disconnected parts of markers (just for finding contours)
            # # warped_t = cv2.morphologyEx(warped_t, cv2.MORPH_CLOSE, np.ones((9, 9)))
            # warped_m = cv2.dilate(warped_t, np.ones((5, 5)), iterations=3)
            # warped_m = cv2.erode(warped_m, np.ones((5, 5)), iterations=3)
            #
            # # remove objects touching the borders
            # warped_c = skiseg.clear_border(warped_m)
            # warped_filled = skimor.remove_small_holes(warped_c, min_size=20000).astype(np.uint8)
            #
            # # plt.figure()
            # # plt.subplot(121), plt.imshow(warped_t, 'gray', interpolation='nearest')
            # # plt.subplot(122), plt.imshow(warped_filled, 'gray', interpolation='nearest')
            # # plt.show()
            #
            # # find individual letters and crop them from image
            # cnts = cv2.findContours(warped_filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            # for c in cnts:
            #     if cv2.contourArea(c) < 200:
            #         continue
            #     (x, y, w, h) = cv2.boundingRect(c)
            #     roi = warped_t[y:y + h, x:x + w]
            #     marker = self.deskew(roi, 20)
            #     marker = self.center_extent(marker, (20, 20))
            #     markers.append((label, marker, (x, y, w, h)))

            markers_sheet = self.detect_markers(warped)
            markers_sheet = [(label, m, r) for m, r in markers_sheet]
            markers.extend(markers_sheet)

            # display rectangle around each marker
            # im_rects = cv2.cvtColor(warped.copy(), cv2.COLOR_GRAY2RGB)
            # for c in cnts:
            #     (x, y, w, h) = cv2.boundingRect(c)
            #     cv2.rectangle(im_rects, (x, y), (x + w, y + h), (0, 255, 0), 4)
            # plt.figure()
            # plt.imshow(im_rects)
            # plt.show()
            # cv2.imshow('markers', im_rects)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return markers

    def recognize(self, img):
        if self.model is not None:
            cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = sorted(cnts, key=cv2.contourArea)[-1]
            (x, y, w, h) = cv2.boundingRect(c)
            roi = img[y:y + h, x:x + w]
            marker = self.deskew(roi, 20)
            marker = self.center_extent(marker, (20, 20))

            prediction = self.model.predict(marker)
        else:
            raise ValueError('No model found - train or load a model first.')
        return prediction


if __name__ == '__main__':
    recog = MarkerRecognizer()

    print 'training ...',
    recog.train_user(markers='SOMC')
    print 'done'

    # save model to disk
    print 'saving model to disk ...',
    joblib.dump(recog.model, '../data/models/model_svm.cpikle')
    print 'done'

    # loading model from disk
    recog.load_model('../data/models/model_svm.cpikle')

    # tests
    print 'testing ...',
    test_sheet = cv2.imread('../data/sheets/test.jpg', 0)
    # test_data = (('test', x) for x in test_sheet)
    sheet = recog.detect_sheet(test_sheet)
    markers = recog.detect_markers(sheet)

    im_vis = cv2.cvtColor(sheet, cv2.COLOR_GRAY2RGB)
    for d, (x, y, w, h) in markers:
        hog = recog.describe(d)
        try:
            label = recog.model.predict(hog)
        except:
            hog = recog.descriptor.describe(d)
            label = recog.model.predict(hog)[0]
            # tmp = cv2.cvtColor(sheet, cv2.COLOR_GRAY2RGB)
            # cv2.rectangle(tmp, (x, y), (x + w, y + h), (0, 255, 0), 4)
            # cv2.imshow('err', tmp)
            # cv2.waitKey(0)

        cv2.rectangle(im_vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(im_vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    print 'done'

    cv2.imshow('test', im_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()