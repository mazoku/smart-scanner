import math

import cv2
import imutils
import numpy as np
import skimage.transform as skitra
import skimage.segmentation as skiseg
import skimage.future as skifut
import skimage.color as skicol
import skimage.feature as skifea
import skimage.morphology as skimor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import helpers
import markerdetector

is_cv_v2 = int(cv2.__version__[0]) == 2


class SheetDetector():
    def __init__(self):
        pass

    def biggest_rot_rect(self, img):
        edges = cv2.Canny(img, 50, 200)
        edges = cv2.dilate(edges, np.ones((3, 3)), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3)), iterations=2)

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
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        chans = cv2.split(img_hsv)

        for i, c in enumerate(chans):
            cs = cv2.bilateralFilter(c, 11, 17, 17)
            # cs = cv2.medianBlur(c, 11)
            chans[i] = cs

        plt.figure()
        plt.subplot(131), plt.imshow(chans[0], 'gray')
        plt.subplot(132), plt.imshow(chans[1], 'gray')
        plt.subplot(133), plt.imshow(chans[2], 'gray')

        colors = ("r", "g", "b")
        plt.figure()
        # loop over the image channels
        for (chan, color) in zip(chans, colors):
            # create a histogram for the current channel and plot it
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.show()
        edges = cv2.Canny(chans[1], 50, 200)

        edges = cv2.dilate(edges, np.ones((3, 3)), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3)), iterations=2)
        cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        biggest_c = cnts[0]
        app = cv2.approxPolyDP(biggest_c, 0.01 * cv2.arcLength(biggest_c, True), True)

        im_vis = img.copy()
        cv2.drawContours(im_vis, [app], -1, (0, 0, 255), 2)
        cv2.imshow('biggest contour', im_vis)
        # cv2.imwrite('biggest_cnt.png', im_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def thresholding(self, img, show=False, show_now=True):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        chans = cv2.split(img_hsv)

        for i, c in enumerate(chans):
            cs = cv2.bilateralFilter(c, 11, 17, 17)
            chans[i] = cs

        candidates = []
        for c in chans:
            ret, th = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # m1 = c < th
            # m2 = c >= th
            m1 = c < ret
            m2 = c >= ret

            for m in (m1, m2):
                cnts = cv2.findContours(m.copy().astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
                cnt = sorted(cnts, key=cv2.contourArea)[-1]
                rect = cv2.minAreaRect(cnt)
                ((x, y), (w, h), rot) = rect
                rat = (w * h) / (c.shape[0] * c.shape[1])

                area = cv2.contourArea(cnt)
                ex = area / (w * h)
                if rat < 0.95:
                    candidates.append((m, ex))

        candidates = sorted(candidates, key=lambda cand: cand[1], reverse=True)

        obj = candidates[0][0].astype(np.uint8)
        # obj = cv2.erode(obj, np.ones((3,3)), iterations=2)
        # obj = cv2.dilate(obj, np.ones((3,3)), iterations=2)

        obj_c = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((3, 3)))
        obj_h = skimor.remove_small_holes(obj_c).astype(np.uint8)
        obj_m = cv2.erode(obj_h, np.ones((5, 5)), iterations=5)
        obj = cv2.dilate(obj_m, np.ones((5, 5)), iterations=5)

        # plt.figure()
        # plt.subplot(141), plt.imshow(obj, 'gray')
        # plt.subplot(142), plt.imshow(obj_c, 'gray')
        # plt.subplot(143), plt.imshow(obj_h, 'gray')
        # plt.subplot(144), plt.imshow(obj_m, 'gray')
        # plt.show()

        cnts = cv2.findContours(obj, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        im_vis = img.copy()
        cv2.drawContours(im_vis, [cnt], -1, (0, 0, 255), 3)

        if show:
            plt.figure()
            plt.subplot(141), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('input'), plt.axis('off')
            plt.subplot(142), plt.imshow(chans[0], 'gray'), plt.title('hue'), plt.axis('off')
            plt.subplot(143), plt.imshow(chans[1], 'gray'), plt.title('saturation'), plt.axis('off')
            plt.subplot(144), plt.imshow(chans[2], 'gray'), plt.title('value'), plt.axis('off')
            # plt.subplot(155), plt.imshow(cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)), plt.title('result'), plt.axis('off')
            if show_now:
                plt.show()

        return cnt

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
        segments = skiseg.slic(img, 30, sigma=0)
        n_segs = segments.max() + 1

        # plt.figure()
        # plt.imshow(skiseg.mark_boundaries(img, segments, (1, 0, 0)))
        # plt.show()

        # defining central roi
        # roi = np.zeros(img.shape[:2])
        # center = (int(round(img.shape[1] / 2)), int(round(img.shape[0] / 2)))
        # cv2.circle(roi, center, radius, 1, -1)
        # roi = roi.astype(np.uint8)

        # labels = [0] * n_segs
        # mask = self.mask2supmask(segments, roi)

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

        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # k-means clustering
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # cv2.imshow('hsv', img_hsv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        segs_cs = []
        for l in range(n_segs):
            # cs_val = [np.median(img_hsv[:, :, i][np.nonzero(segments == l)]) for i in range(3)]
            cs_val = [np.median(img_cs[:, :, i][np.nonzero(segments == l)]) for i in range(3)]
            # cs_val = [img_cs[:, :, i][np.nonzero(segments == l)].mean() for i in range(3)]
            segs_cs.append(cs_val)

        clt = KMeans(n_clusters=2)
        clt.fit(segs_cs)

        res = np.zeros(img.shape[:2])
        for l in range(n_segs):
            res[np.nonzero(segments == l)] = clt.labels_[l]
        res = res.astype(np.uint8)

        cnt = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2][0]
        app = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        im_vis = img.copy()
        # cv2.drawContours(im_vis, [app], -1, (0, 0, 255), 2)
        cv2.drawContours(im_vis, [cnt], -1, (0, 0, 255), 2)

        cv2.imshow('clustering', res)
        cv2.imshow('sheet', im_vis)
        # cv2.imwrite('kmeans.png', im_vis)

        print ['{}: {}'.format(i, c) for i, c in enumerate(segs_cs)]
        plt.figure()
        plt.subplot(221), plt.imshow(skiseg.mark_boundaries(img, segments, (1, 0, 0)))
        plt.subplot(222), plt.imshow(segments, 'jet')
        plt.subplot(223), plt.imshow(im_vis)
        plt.subplot(224), plt.imshow(segments == 2)
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return app

    def grow_mask(self, lab_im, mask, img, t=20):
        #TODO: konverzi obrazku do HSV udelat pouze jednou vne teto metody
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

    def mser(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gauss = cv2.GaussianBlur(gray, (5, 5), 0)
        bil = cv2.bilateralFilter(gray, 0, sigmaColor=5, sigmaSpace=10)

        # detect MSER keypoints in the image
        detector = cv2.MSER_create(1, 1000, 94400, 0.05, 0.02, 200, 1.01, 0.003, 5)
        regions = detector.detectRegions(bil, None)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        im_vis = img.copy()

        cv2.polylines(im_vis, hulls, 1, (0, 255, 0))
        # loop over the keypoints and draw them
        # for kp in kps:
        #     r = int(0.5 * kp.size)
        #     (x, y) = np.int0(kp.pt)
        #     cv2.circle(im_vis, (x, y), r, (0, 255, 255), 2)

        # show the image
        cv2.imshow("Images", np.hstack([img, im_vis]))
        cv2.waitKey(0)

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
            roi = markerdetector.get_roi(pt, img.shape, width=31, height=31, img=img)
            markerdetector.detect_corner(img, roi)

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

    def _weight_mean_color(self, graph, src, dst, n):
        """Callback to handle merging nodes by recomputing mean color.

        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """

        diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        diff = np.linalg.norm(diff)
        return {'weight': diff}

    def merge_mean_color(self, graph, src, dst):
        """Callback called before merging two nodes of a mean color distance graph.

        This method computes the mean color of `dst`.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        """
        graph.node[dst]['total color'] += graph.node[src]['total color']
        graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
        graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                         graph.node[dst]['pixel count'])
    def rag(self, img):
        # orig = img.copy()
        labels = skiseg.slic(img, compactness=30, n_segments=400)
        g = skifut.graph.rag_mean_color(img, labels)

        labels2 = skifut.graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=self.merge_mean_color,
                                           weight_func=self._weight_mean_color)

        g2 = skifut.graph.rag_mean_color(img, labels2)

        out = skicol.label2rgb(labels2, img, kind='avg')
        out = skiseg.mark_boundaries(out, labels2, (0, 0, 0))

        cv2.imshow('in', img)
        cv2.imshow('RAG', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect(self, img, show=False, show_now=True):
        # compute the scale for resizing
        if img.shape[0] > img.shape[1]:
            scale = img.shape[1] // 500
        else:
            scale = img.shape[0] // 500

        orig = img.copy()
        img = cv2.resize(img, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_AREA)

        # convert the image to grayscale, blur it, and find edges
        # in the image
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gauss = cv2.GaussianBlur(gray, (5, 5), 0)
        # bil = cv2.bilateralFilter(gray, 0, sigmaColor=5, sigmaSpace=10)

        # cv2.imshow('gauss', gray)
        # cv2.imshow('bil', bil)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # biggest rot rect
        # self.biggest_rot_rect(img)

        # biggest contour
        # self.biggest_contour(img)

        # probabilistic hough
        # self.lines(img, edges)

        # thresholding
        cnt = self.thresholding(img, show=False)

        # color model
        # self.color_model(img)

        # superpixels
        # cnt = self.superpixels(img)

        # mser
        # self.mser(img)

        # RAG
        # self.rag(img)

        # resize to original size
        cnt *= scale

        # align image
        warped = helpers.four_point_transform(orig, cnt.reshape(4, 2))

        img_vis = orig.copy()
        cv2.drawContours(img_vis, [cnt], -1, (0, 0, 255), 8)

        if show:
            plt.figure()
            plt.subplot(121), plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            plt.subplot(122), plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            if show_now:
                plt.show()

        # cv2.imshow('orig', img_vis)
        # cv2.imshow('warped', warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('notes.png', warped)

        # img_vis = img.copy()
        # cv2.drawContours(img_vis, [cnt], -1, (0, 0, 255), 3)
        return img_vis, warped

if __name__ == '__main__':
    fname1 = '../data/scan1.jpg'
    fname2 = '../data/scan2.jpg'

    fnames = (fname1, fname2)
    # fnames = (fname2,)
    res = []
    warps = []
    for f in fnames:
        im = cv2.imread(f)

        detector = SheetDetector()
        im_res, warped = detector.detect(im)
        res.append(im_res)
        warps.append(warped)

    plt.figure()
    for i, r in enumerate(warps):
        cv2.imwrite('../data/sheet%i.png' % (i + 1), r)
        plt.subplot(1, len(warps), i + 1)
        plt.imshow(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()