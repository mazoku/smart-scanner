import cv2
import imutils
import numpy as np
import skimage.transform as skitra

import helpers

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
        ul = inters[np.argmin(sums)]
        br = inters[np.argmax(sums)]

        img_vis = img.copy()
        for l in line_types:
            cv2.line(img_vis, l[0], l[1], (255, 0, 0), 2)
        for pt in inters:
            cv2.circle(img_vis, pt, 5, (0, 255, 0), 2)
        cv2.circle(img_vis, ul, 5, (255, 0, 255), 2)
        cv2.circle(img_vis, br, 5, (255, 0, 255), 2)
        cv2.imshow('expanded', img_vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

        # probabilistic hough
        self.lines(img, edges)

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