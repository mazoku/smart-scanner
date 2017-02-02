import math

import cv2
import imutils
import numpy as np
import skimage.transform as skitra
import skimage.segmentation as skiseg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import helpers
import corner_detector

is_cv_v2 = int(cv2.__version__[0]) == 2


class SheetDetector():
    def __init__(self):
        pass

    def biggest_rot_rect(self, img, edges):
        # vybrat nejvetsi konturu
        cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        biggest_c = cnts[0]

        im_vis = img.copy()
        cv2.drawContours(im_vis, [biggest_c], -1, (255, 0, 255), 4)

        # rotated bounding box
        rotrect = cv2.minAreaRect(biggest_c)
        if is_cv_v2:
            box = cv2.cv.BoxPoints(rotrect)
        else:
            box = cv2.boxPoints(rotrect)
        box = np.int0(box)
        cv2.drawContours(im_vis, [box], 0, (0, 0, 255), 2)

        cv2.imshow('Biggest contour', im_vis)

    def biggest_contour(self, img):
        edges = cv2.Canny(img, 50, 200)

        cv2.imshow('edges', edges)
        cv2.waitKey(0)

        edges = cv2.dilate(edges, np.ones((3, 3)), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3)), iterations=2)
        cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        biggest_c = cnts[0]
        app = cv2.approxPolyDP(biggest_c, 0.01 * cv2.arcLength(biggest_c, True), True)

        im_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(im_vis, [app], -1, (0, 0, 255), 2)
        cv2.imshow('biggest contour', im_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def threshold(self, img):
        ret, th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        th = cv2.erode(th, np.ones((3, 3)), iterations=2)
        th = cv2.dilate(th, np.ones((3, 3)), iterations=2)

        cnts = cv2.findContours(th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        biggest_c = cnts[0]
        app = cv2.approxPolyDP(biggest_c, 0.01 * cv2.arcLength(biggest_c, True), True)

        # im_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        im_vis = img.copy()
        cv2.drawContours(im_vis, [app], -1, (0, 0, 255), 2)
        cv2.imshow('biggest contour', im_vis)

        cv2.imshow('otsu', th)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

    def color_model(self, img, radius=80, width=10):
        in_roi = np.zeros(img.shape[:2])
        center = (int(round(img.shape[1] / 2)), int(round(img.shape[0] / 2)))
        cv2.circle(in_roi, center, radius, 1, -1)
        in_roi = in_roi.astype(np.uint8)

        out_roi = np.ones(img.shape[:2], dtype=np.uint8)
        out_roi[width:out_roi.shape[0] - width, width:out_roi.shape[1] - width] = 0

        im_rois = img.copy()
        in_roi = np.dstack((np.zeros_like(in_roi), 255 * in_roi, np.zeros_like(in_roi)))
        out_roi = np.dstack((np.zeros_like(out_roi), np.zeros_like(out_roi), 255 * out_roi))
        im_rois = cv2.addWeighted(im_rois, 1, in_roi, 0.5, 1)
        im_rois = cv2.addWeighted(im_rois, 1, out_roi, 0.5, 1)

        # cv2.imshow('rois', in_roi + out_roi)
        cv2.imshow('rois', im_rois)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def superpixels(self, img, radius=20):
        segments = skiseg.slic(img, 30, sigma=1)
        n_segs = segments.max() + 1

        # defining central roi
        roi = np.zeros(img.shape[:2])
        center = (int(round(img.shape[1] / 2)), int(round(img.shape[0] / 2)))
        cv2.circle(roi, center, radius, 1, -1)
        roi = roi.astype(np.uint8)

        # labels = [0] * n_segs
        mask = self.mask2supmask(segments, roi)

        # region growing
        # change = True
        # while change:
        #     change, mask_out = self.grow_mask(segments, mask, img)
        #
        #     plt.figure()
        #     plt.subplot(221), plt.imshow(skiseg.mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), segments), 'gray')
        #     plt.subplot(222), plt.imshow(segments)
        #     plt.subplot(223), plt.imshow(mask, 'gray')
        #     plt.subplot(224), plt.imshow(mask_out, 'gray')
        #     plt.show()
        #     mask = mask_out

        # k-means clustering
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        segs_hsv = []
        for l in range(n_segs):
            hsv_val = [np.median(img_hsv[:, :, i][np.nonzero(segments == l)]) for i in range(3)]
            segs_hsv.append(hsv_val)

        clt = KMeans(n_clusters=2)
        clt.fit(segs_hsv)

        res = np.zeros(img.shape[:2])
        for l in range(n_segs):
            res[np.nonzero(segments == l)] = clt.labels_[l]
        res = res.astype(np.uint8)

        cnt = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2][0]
        app = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        im_vis = img.copy()
        cv2.drawContours(im_vis, [app], -1, (0, 0, 255), 2)

        cv2.imshow('clustering', res)
        cv2.imshow('sheet', im_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(skiseg.mark_boundaries(img, segments))
        # plt.axis("off")
        # plt.show()

    def grow_mask(self, lab_im, mask, img, t=20):
        #TODO: konverzi obrazku do HDV udelat poue jednou vne teto metody
        change = False
        mask_out = mask.copy()
        # find neighboring superpixels
        labs = self.get_nghbs(lab_im, mask)

        # check whether the superpixel should be attached to the mask
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #29, 38
        # s1 = lab_im == 29
        # s2 = lab_im == 38
        #
        # bgr1 = [img[:, :, i][np.nonzero(s1)].mean() for i in range(3)]
        # bgr2 = [img[:, :, i][np.nonzero(s2)].mean() for i in range(3)]
        #
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv1 = [img_hsv[:, :, i][np.nonzero(s1)].mean() for i in range(3)]
        # hsv2 = [img_hsv[:, :, i][np.nonzero(s2)].mean() for i in range(3)]
        #
        # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # lab1 = [img_lab[:, :, i][np.nonzero(s1)].mean() for i in range(3)]
        # lab2 = [img_lab[:, :, i][np.nonzero(s2)].mean() for i in range(3)]
        #
        # print 'BGR:', bgr1, bgr2, ', dist:', np.linalg.norm([x - y for x, y in zip(bgr1, bgr2)])
        # print 'HSV:', hsv1, hsv2, ', dist:', np.linalg.norm([x - y for x, y in zip(hsv1, hsv2)])
        # print 'LAB:', lab1, lab2, ', dist:', np.linalg.norm([x - y for x, y in zip(lab1, lab2)])

        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()
        # mask_med = np.median(gray[np.nonzero(mask)])
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_val = [np.median(img_hsv[:, :, i][np.nonzero(mask)]) for i in range(3)]
        for l in labs:
            # seg_med = np.median(gray[np.nonzero(lab_im == l)])
            # seg_mean = [img_hsv[:, :, i][np.nonzero(lab_im == l)].mean() for i in range(3)]
            seg_val = [np.median(img_hsv[:, :, i][np.nonzero(lab_im == l)]) for i in range(3)]
            # if abs(mask_med - seg_med) < t:
            dist = np.linalg.norm([x - y for x, y in zip(mask_val, seg_val)])
            print 'mask={}, label ={}, seg={}, dist={}'.format(mask_val, l, seg_val, dist)
            if dist < t:
                change = True
                mask_out += lab_im == l

        return change, mask_out

    def get_nghbs(self, lab_im, mask):
        mask_dil = cv2.dilate(mask, np.ones((3, 3))) - mask
        labs = lab_im[np.nonzero(mask_dil)]
        labs = np.unique(labs)
        return labs

    def mask2supmask(self, lab_im, mask):
        '''
        Creates a mask that respects the superpixels
        :param lab_im: superpixels
        :param mask: general mask to be updated
        :return: mask that respects the superpixels
        '''
        mask_out = np.zeros(mask.shape, dtype=np.uint8)
        labs = lab_im[np.nonzero(mask)]
        labs = np.unique(labs)
        for l in labs:
            mask_out += lab_im == l
        return mask_out

    def lines(self, img, edges):
        lines = skitra.probabilistic_hough_line(edges, threshold=10, line_length=80, line_gap=2)

        # sort the lines by their length (descending order)
        lines = sorted(lines, key=helpers.line_length, reverse=True)
        # filter out small line segments
        lines = [l for l in lines if helpers.line_length(l) > 0.5 * helpers.line_length(lines[0])]

        # longest = lines[0]
        # positions = [helpers.line_position(longest, l) for l in lines[1:]]
        # perps = [lines[i] for i, p in enumerate(positions) if p == 'perp']
        # paras = [lines[i] for i, p in enumerate(positions) if p == 'para']

        line_types = helpers.merge_lines(lines, show=False)
        x_max = img.shape[1]
        y_max = img.shape[0]
        line_types = [helpers.expand_line(l[0], l[1], (0, x_max), (0, y_max)) for l in line_types]

        inters = []
        for i, l1 in enumerate(line_types):
            for l2 in line_types[i + 1:]:
                inter = tuple(helpers.line_intersect(l1, l2).astype(np.int))
                if helpers.in_img(inter, img.shape):
                    inters.append(inter)

        sums = [x + y for x, y in inters]
        diffs = [x - y for x, y in inters]
        tl = inters[np.argmin(sums)]
        br = inters[np.argmax(sums)]
        tr = inters[np.argmax(diffs)]
        bl = inters[np.argmin(diffs)]
        corners = (tl, tr, br, bl)
        sheet = np.array([np.expand_dims(np.array(x), 0) for x in corners])

        img_vis = img.copy()
        for l in line_types:
            cv2.line(img_vis, l[0], l[1], (255, 0, 0), 2)
        cv2.drawContours(img_vis, [sheet], -1, (255, 255, 0), 2)
        for pt in inters:
            cv2.circle(img_vis, pt, 5, (0, 255, 0), 2)
        texts = ('tl', 'tr', 'br', 'bl')
        for pt, txt in zip(corners, texts):
            cv2.circle(img_vis, pt, 5, (255, 0, 255), 2)
            cv2.putText(img_vis, txt, (pt[0], pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.imshow('expanded', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for pt in corners:
            roi = corner_detector.get_roi(pt, img.shape, width=31, height=31, img=img)
            corner_detector.detect_corner(img, roi)

        # inters = []
        # line_types = helpers.merge_lines(lines)
        # for i, l1 in enumerate(line_types):
        #     for l2 in line_types[i + 1:]:
        #         inters.append(helpers.line_intersect(l1, l2))
        # # inters = [helpers.line_intersect(l1, l2) for l1 in line_types for l2 in line_types[1:]]
        # sums = [x + y for x, y in inters]
        # ul = inters[np.argmin(sums)]
        # br = inters[np.argmax(sums)]
        #
        # im_pts = img.copy()
        # cv2.circle(im_pts, ul, 5, (0, 0, 255), 2)
        # cv2.circle(im_pts, br, 5, (255, 0, 0), 2)
        # cv2.imshow('points', im_pts)

        # for l, p in zip(lines[1:], positions):
        #     im_vis = img.copy()
        #     cv2.line(im_vis, longest[0], longest[1], (255, 0, 0), 2)
        #     cv2.line(im_vis, l[0], l[1], (0, 0, 255), 2)
        #     cv2.putText(im_vis, str(p), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     cv2.imshow('line', im_vis)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # visualization
        # im_hough = img.copy()
        # for l in lines:
        #     cv2.line(im_hough, l[0], l[1], (0, 0, 255), 2)
        #
        # cv2.imshow('hough', im_hough)

    def detect(self, img):
        # compute the ratio of the old height to the new height, clone it, and resize it
        ratio = img.shape[0] / 500.0
        orig = img.copy()
        img = imutils.resize(img, height=500)

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gauss = cv2.GaussianBlur(gray, (5, 5), 0)
        bil = cv2.bilateralFilter(gray, 0, sigmaColor=5, sigmaSpace=10)

        # cv2.imshow('gauss', gray)
        # cv2.imshow('bil', bil)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # TODO: pridat okraje pro analyzu jasu listu a pozadi
        # pridat region uprostred, ktery musi obsahovat list
        # pridat okraje, ktere nesmi obsahovat list
        edges = cv2.Canny(bil, 50, 200)
        edges = cv2.dilate(edges, np.ones((3, 3)), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3)), iterations=2)

        # biggest rot rect
        # self.biggest_rot_rect(img, edges)

        # biggest contour
        # self.biggest_contour(bil)

        # probabilistic hough
        # self.lines(img, edges)

        # thresholding
        # self.threshold(img)

        # color model
        # self.color_model(img)

        # superpixels
        self.superpixels(img)

        # show the original image and the edge detected image
        # cv2.imshow("Image", img)
        # cv2.imshow("Edged", edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    fname = '../data/sheet1.png'
    im = cv2.imread(fname)

    sd = SheetDetector()
    sd.detect(im)