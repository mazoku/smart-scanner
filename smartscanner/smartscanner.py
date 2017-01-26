import cv2

from sheet import SheetDetector

# read test image
fname = '../data/sheet1.png'
im = cv2.imread(fname)

sd = SheetDetector()