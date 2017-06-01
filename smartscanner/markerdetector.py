from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt

import skimage.transform as skitra
import skimage.segmentation as skiseg


class MarkerDetector():
    def __init__(self):
        pass

    def contour_on_border(self, cnt, shape):
        border = np.zeros(shape)
        border[0, :] = 1
        border[-1, :] = 1
        border[:, 0] = 1
        border[:, -1] = 1
        im = np.zeros(shape)
        cv2.drawContours(im, [cnt], -1, 1, 2)
        inters = cv2.bitwise_and(border, im)

        # plt.figure()
        # plt.suptitle(inters.sum())
        # plt.subplot(131), plt.imshow(border, 'gray', interpolation='nearest')
        # plt.subplot(132), plt.imshow(im, 'gray', interpolation='nearest')
        # plt.subplot(133), plt.imshow(inters, 'gray', interpolation='nearest')
        # plt.show()

        if inters.sum() > 0:
            return True
        else:
            return False

    def detect(self, im, part_x=1./4, part_y=1./8):
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        end_r = int(img.shape[0] * part_y)
        end_c = int(img.shape[1] * part_x)
        band = img[:end_r, :end_c]

        band_s = cv2.bilateralFilter(band, 11, 17, 17)
        th, band_t = cv2.threshold(band_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts = cv2.findContours(band_t.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        ext_max = 0.8
        marker_cnt = None
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            extent = cv2.contourArea(c) / float(w * h)
            if extent > ext_max:
                if not self.contour_on_border(c, band_t.shape):
                    marker_cnt = c
                    ext_max = extent

        #     tmp = im_vis.copy()
        #     cv2.drawContours(tmp, [c], -1, (0, 0, 255), 2)
        #     cv2.imshow('{:.2f}, {}'.format(extent, self.contour_on_border(c, band_t.shape)), tmp)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # TODO: vyriznout marker
        #   -  oriznout podle bounding rectu kontury
        #   -  naprahovat
        #   -  odstranit vsechny cerne objekty dotykajici se hrany
        (x, y, w, h) = cv2.boundingRect(marker_cnt)
        marker = band_s[y:y + h, x:x + w]
        th, _ = cv2.threshold(marker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        marker_t = (marker < th).astype(np.uint8)
        marker = skiseg.clear_border(marker_t)

        # visualization
        band_vis = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(band_vis, [marker_cnt], -1, (0, 0, 255), 2)
        cv2.imshow('band', band_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imshow('marker', 255 * marker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite('../data/band.jpg', band_vis)
        cv2.imwrite('../data/marker.jpg', 255 * marker)

        return marker_cnt


if __name__ == '__main__':
    fname1 = '../data/sheet3.png'
    fname2 = '../data/sheet4.png'

    fnames = (fname1, fname2)
    # fnames = (fname2,)
    marks = []
    for f in fnames:
        im = cv2.imread(f)
        detector = MarkerDetector()

        marker = detector.detect(im)