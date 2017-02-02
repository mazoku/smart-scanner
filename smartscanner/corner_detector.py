from __future__ import division

import numpy as np
import cv2
from skimage.feature import corner_harris, corner_peaks


def detect_corner(img, roi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # ret, th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    cv2.imshow('otsu', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_c = img[roi[0, 0, 1]:roi[2, 0, 1] + 1, roi[0, 0, 0]:roi[2, 0, 0] + 1, :]

    # cv2.imshow('orig', img)
    # cv2.imshow('crop', img_c)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return

    # coords = corner_peaks(corner_harris(img), min_distance=5)
    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    ret, th_o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # img_m = cv2.medianBlur(gray, 5)
    # th_a = cv2.adaptiveThreshold(img_m, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    cv2.imshow('orig', img_c)
    cv2.imshow('otsu', th_o)
    # cv2.imshow('adaptive', th_a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # harris
    # gray_f = np.float32(gray)
    # dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    # # Threshold for an optimal value, it may vary depending on the image.
    # # img_c[dst > 0.01 * dst.max()] = [0, 0, 255]
    # img_c[dst == dst.max()] = [0, 0, 255]
    # cv2.imshow('harris', img_c)

    # fast
    # # Initiate FAST object with default values
    # fast = cv2.FastFeatureDetector()
    # # find and draw the keypoints
    # kp = fast.detect(img_c, None)
    # img2 = cv2.drawKeypoints(img_c, kp, color=(255, 0, 0))
    # cv2.imshow('fast', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_roi(pt, shape, width=50, height=50, img=None):
    '''

    :param pt: center of the roi, (x, y)
    :param shape: image shape (rows, cols)
    :param width:
    :param height:
    :return:
    '''
    roi = np.zeros(shape, dtype=np.bool)

    # height and width need to be odd number
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    start_r = int(pt[1] - height / 2)
    end_r = start_r + height
    start_c = int(pt[0] - width / 2)
    end_c = start_c + width

    roi[start_r:end_r + 1, start_c:end_c + 1] = 1
    pts = ((start_c, start_r), (end_c, start_r), (end_c, end_r), (start_c, end_r))
    roi_cnt = np.array([np.expand_dims(p, 0) for p in pts])

    if img is None:
        im_vis = np.zeros(shape)
    else:
        im_vis = img.copy()
    cv2.circle(im_vis, pt, 5, (0, 0, 255), 2)
    cv2.drawContours(im_vis, [roi_cnt], -1, (255, 0, 255), 0)
    # cv2.imshow('pt roi', im_vis)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    return roi_cnt