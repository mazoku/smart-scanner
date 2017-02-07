from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt

class MarkerDetector():
    def __init__(self):
        pass

    def detect(self, im, partition=1./8):
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        end_r = int(img.shape[0] * partition)
        band = img[:end_r, :]

        # cv2.imshow('band', band)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        scale = int(band.shape[1] / 400)
        band_r = cv2.resize(band, None, fx=1./scale, fy=1./scale)
        th, band_t = cv2.threshold(band_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        projection_to_x = band_t.sum(axis=0)
        projection_to_y = band_t.sum(axis=1)

        plt.figure()
        plt.subplot(121), plt.plot(projection_to_x, 'b-')
        plt.subplot(122), plt.plot(projection_to_y, 'b-')
        plt.show()


if __name__ == '__main__':
    fname1 = '../data/sheet1.png'
    fname2 = '../data/sheet2.png'

    fnames = (fname1, fname2)
    # fnames = (fname2,)
    marks = []
    for f in fnames:
        im = cv2.imread(f)
        detector = MarkerDetector()

        marker = detector.detect(im)