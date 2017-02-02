import cv2

from sheetdetector import SheetDetector

# read test image
fname = '../data/sheet1.png'
im = cv2.imread(fname)

sd = SheetDetector()