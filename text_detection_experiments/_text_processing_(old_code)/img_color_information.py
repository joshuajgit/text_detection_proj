import cv2 as cv
import numpy as np
import math
import time

from statistics import mode

# invert the colors in this grayscale image
def invert_colors(image):
    return cv.bitwise_not(image)

# get darkest and lightest colors, particularly in grayscale;
# scale factor used to shrink image for sake of optimization
# if not included then defaults to 1
def min_max_colors(image, scale_factor=1):
    (h, w) = image.shape[:2]
    img_copy = cv.resize(image, (int(w*scale_factor), int(h*scale_factor)), cv.INTER_AREA)    
    (h_prime, w_prime) = img_copy.shape[:2]
    
    img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    all_colors = []

    for r in range(0, w_prime):
        for c in range (0, h_prime):
            all_colors.append(img_copy[c][r])

    darkest = min(all_colors)
    lightest = max(all_colors)

    return darkest, lightest, img_copy

# get the most commonly occurring color in a grayscale image
# scale factor used to shrink image for sake of optimization
# if not included then defaults to 1
def dominant_color(image, scale_factor=1):
    (h, w) = image.shape[:2]
    img_copy = cv.resize(image, (int(w*scale_factor), int(h*scale_factor)), cv.INTER_AREA)    
    (h_prime, w_prime) = img_copy.shape[:2]
    
    all_colors = []

    for r in range(0, w_prime):
        for c in range (0, h_prime):
            all_colors.append(img_copy[c][r])

    dominant_col = mode(all_colors) # <- get most frequently occurring color

    return dominant_col, img_copy

# when applying a lot of padding, the words next to
# the word being processed can show up in the crop at the edges.
# this function gets rid of the words on the sides.
def clean_edges(image, index):
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    mask.fill(255) # <- make mask image entirely white to begin with

    orig = image
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    (h, w) = image.shape[:2]
    (h_orig, w_orig) = image.shape[:2]
    image = invert_colors(image)

    dfw, dfh = 15, 1
    dilation_kernel = np.ones((dfh,dfw), np.uint8)
    image = cv.dilate(image, dilation_kernel, iterations=2)

    cv.namedWindow("dilated_{0} (clean_edges)".format(index), cv.WINDOW_AUTOSIZE )
    cv.imshow("dilated_{0} (clean_edges)".format(index), image)
    
    # get a contour shaped by the text; image must be white text
    # on black background
    contours, hierarchy = cv.findContours(image,\
        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rects = []
    rois = []
    for i in range(len(contours)):
        x_vals = []
        y_vals = []
        cnt = contours[i]
        for j in range(0, len(cnt)):
            x_vals.append(cnt[j][0][0])
            y_vals.append(cnt[j][0][1])
        
        min_x, max_x, min_y, max_y = \
            min(x_vals), max(x_vals), min(y_vals), max(y_vals)
        start_pt, end_pt = (min_x, min_y), (max_x, max_y)

        rects.append((start_pt, end_pt))

        roi = orig[min_y:max_y, min_x:max_x]
        rois.append(roi)

    _index = 0
    biggest_rect_index = -1 # <- index corresponding to the biggest rectangle we have
    biggest_rect_val = 0
    final_startX = -1
    final_startY = -1
    final_w = -1
    final_h = -1
    
    for rect in rects:
        startX = rect[0][0]
        startY = rect[0][1]
        endX = rect[1][0]
        endY = rect[1][1]
        w, h = endX-startX, endY-startY

        if (w*h > biggest_rect_val):
            biggest_rect_val = w*h
            biggest_rect_index = _index
            final_w, final_h = w, h
            final_startX, final_startY = startX, startY

        _index += 1

    roi_keep = rois[biggest_rect_index]    
    mask[startY:startY+final_h,final_startX:final_startX+final_w] = roi_keep    
    image = mask
    
    return image
