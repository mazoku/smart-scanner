import cv2
import imutils
import skimage.restoration as skires

class SheetDetector():
    def __init__(self):
        pass

    def detect(self, image):
        # compute the ratio of the old height to the new height, clone it, and resize it
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = imutils.resize(image, height=500)

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gauss = cv2.GaussianBlur(gray, (5, 5), 0)
        bil = cv2.bilateralFilter(gray, 0, sigmaColor=5, sigmaSpace=10)

        # cv2.imshow('gauss', gray)
        # cv2.imshow('bil', bil)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # TODO: pridat okraje pro analyzu jasu listu a pozadi
        # pridat region uprostred, ktery musi obsahovat list
        # pridat okraje, ktere nesmi obsahovat list
        edged = cv2.Canny(bil, 50, 200)

        # show the original image and the edge detected image
        print "STEP 1: Edge Detection"
        cv2.imshow("Image", image)
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fname = '../../data/sheet1.png'
    im = cv2.imread(fname)

    sd = SheetDetector()
    sd.detect(im)