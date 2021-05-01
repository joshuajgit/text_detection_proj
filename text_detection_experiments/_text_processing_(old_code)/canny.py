import cv2 as cv
import numpy as np

import sys
sys.path.append('_text_processing')

import img_color_information
import skew_correction

def get_canny(image, edge_type):
    # apply automatic Canny edge detection using the computed median
    # (Adrian Rosebrock, 2015)
    # https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-
    # canny-edge-detection-with-python-and-opencv/

    if edge_type == "AUTO":
        # AUTO
        v = np.median(image)
        sigma = 0.33
        
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        print ("Canny: <{0}, {1}>".format(lower, upper))

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, lower, upper)

    if edge_type == "TIGHT":
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 200, 250)

    cv.namedWindow("Canny", cv.WINDOW_AUTOSIZE)
    cv.imshow("Canny", canny)

    return canny


