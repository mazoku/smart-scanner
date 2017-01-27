import cv2
import numpy as np
import imutils
import skimage.restoration as skires
import skimage.transform as skitra

is_cv_v2 = int(cv2.__version__[0]) == 2


class SheetDetector():
    def __init__(self):
        pass

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

        # probabilistic hough
        lines = skitra.probabilistic_hough_line(edges, threshold=10, line_length=80, line_gap=2)
        im_hough = img.copy()
        for l in lines:
            cv2.line(im_hough, l[0], l[1], (0, 0, 255), 2)
        # show the original image and the edge detected image
        cv2.imshow("Image", img)
        cv2.imshow("Edged", edges)
        cv2.imshow('Biggest contour', im_vis)
        cv2.imshow('hough', im_hough)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fname = '../../data/sheet1.png'
    im = cv2.imread(fname)

    sd = SheetDetector()
    sd.detect(im)