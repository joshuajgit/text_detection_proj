import text_recognition
import imutils
import cv2 as cv
import numpy as np
import skew_correction
import pytesseract

import google_translate

# invert the colors in this grayscale image
def invert_colors(image, h, w):
    return cv.bitwise_not(image)

def cluster(image, all_rois, rect_info, avg_roi_height, rW, rH, p):
    show_rois = False
    verbose = False

    texts_found = []
    cluster_rects = []
    text_locs = [] # <- locations of each roi corresponding to text

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    mask = np.zeros(image.shape, np.uint8)
    mask.fill(255) # <- make mask image entirely white to begin with

    (H, W) = mask.shape[:2]

    if (verbose):
        print ("rect info: {0}".format(rect_info))

    counter = 0
    for (x, y, w, h) in rect_info:
        if (verbose):
            print ("{0}, {1}, {2}, {3}".format(x, y, w, h))
        startX = int(x-((p/100)*x))
        endX = int(x+w+((p/100)*(x+w)))
        startY = int(y-((p/100)*y))
        endY = int(y+h+((p/100)*(y+h)))
        if (startX < 0):
            startX = 0
        if (endX > W-1):
            endX = W-1
        if (startY < 0):
            startY = 0
        if (endY > H-1):
            endY = H-1
        
        if (len(all_rois[counter].shape) == 3):
            for k in range (0, 2):
                all_rois[counter] = np.delete(all_rois[counter], 0, 2)
            all_rois[counter] = np.squeeze(all_rois[counter])

        if (verbose):
            print ("mask: {0}".format(mask[startY:endY,startX:endX].shape))
            print ("roi: {0}".format(all_rois[counter].shape))

        H, W = all_rois[counter].shape

        mask[startY:startY+H,startX:startX+W] = all_rois[counter]
        
        counter += 1

    image = mask

    # Idea 1: clustering with image dilation
    df = int(avg_roi_height / 3)
    dilation_kernel = np.ones((df,df), np.uint8)
    
    (h, w) = image.shape[:2]
    image = invert_colors(image, h, w)

    if (show_rois):
        cv.namedWindow("Before dilate (cluster())", cv.WINDOW_AUTOSIZE )
        cv.imshow("Before dilate (cluster())", image)
    
    image = cv.dilate(image, dilation_kernel, iterations=2)
    
    # get a contour shaped by the text; image must be white text
    # on black background
    contours, hierarchy = cv.findContours(image,\
        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # reads text detections as uniform blocks of text
    tess_config = ("-l eng --oem 1 --psm 6")
    
    if (show_rois):
        cv.namedWindow("Text recog window (cluster())", cv.WINDOW_AUTOSIZE )
        cv.imshow("Text recog window (cluster())", image)

    rects = []
    rois = []
    for i in range(len(contours)):
        x_vals = []
        y_vals = []
        cnt = contours[i]
        for j in range(0, len(cnt)):
            x_vals.append(cnt[j][0][0])
            y_vals.append(cnt[j][0][1])
        
        min_x = min(x_vals)
        max_x = max(x_vals)
        min_y = min(y_vals)
        max_y = max(y_vals)

        start_pt = (min_x, min_y)
        end_pt = (max_x, max_y)

        print ("START PT: {0}".format(start_pt))
        print ("END PT: {0}".format(end_pt))

        rects.append((start_pt, end_pt))
        image = cv.rectangle(image, start_pt, end_pt, (0, 255, 0), 2)

        roi = mask[min_y:max_y, min_x:max_x]
        rois.append(roi)
        cluster_rects.append((min_x, min_y, max_x, max_y))

    for i in range(0, len(rois)):
        if (show_rois):
            cv.namedWindow("ROI #{0} (cluster())", cv.WINDOW_AUTOSIZE )
            cv.imshow("ROI #{0} (cluster())", rois[i])
        
        text = pytesseract.image_to_string(rois[i], config=tess_config)
        
        # remove any non-ASCII characters that may be picked
        # up by mistake, + strip some other characters that aren't letters
        text = "".join(j for j in text if ord(j) < 128).strip()
        text = text.strip("/\|")
        
        text = text.replace("\n", " ") # <- get rid of new line characters        
        print ("PSM 6 result: {0}".format(text))

        texts_found.append(text)

    print (texts_found)        
    return texts_found, cluster_rects
