from imutils.object_detection import non_max_suppression
from nms import nms

import cv2 as cv
import numpy as np
import statistics
import imutils
import pytesseract
import text_utils

import skew_correction
import img_color_information
import cluster_texts

import time
import threading

# other scripts written to assist in image preprocessing

east_path = "learning_models/text_processing/frozen_east_text_detection.pb"

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

pytesseract.pytesseract.tesseract_cmd = \
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# default values for east blob size
east_w = 320
east_h = 320

# confidence level required to be met to say a given image has text
min_confidence = 0.99
padding = 2

counter = 0
texts_found = []

# the calls to Tesseract that each thread makes are here
def tesseract_func(roi_rotated, counter):
    global texts_found
    
    config = ("-l eng --oem 1 --psm 6")
    config = ("-l eng --oem 1 --psm 8")
    text = pytesseract.image_to_string(roi_rotated, config=config)
    print ("TEXT #{0}: {1}\n".format(counter, text))

    # remove any non-ASCII characters that may be picked
    # up by mistake
    text = ''.join(i for i in text if ord(i) < 128).strip()

    texts_found.append((counter, text))

# main text recognition function
def text_recognition(img, net, mode):
    # return_data may collect text detections as words or clusters of
    # words, or individual characters
    
    show_rois = True
    verbose = False
    show_polys = False
    do_draw = True

    global counter
    global texts_found

    # track time to test algorithm's speed...
    start = time.time()

    orig, draw_on = img.copy(), img.copy()
    (H, W) = img.shape[:2]
    (orig_H, orig_W) = img.shape[:2]

    newW, newH = (east_w, east_h)
    rW = W / float(newW)
    rH = H / float(newH)

    img = cv.resize(img, (newW, newH))
    (H, W) = img.shape[:2]

    blob = cv.dnn.blobFromImage(img, 1.0, (newW, newH),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)

    print ("Orig image dims: {0}x{1}".format(orig_W, orig_H))

    (rects, confidences, baggage) = decode(scores, geometry)
    
    if (len(rects) == 0): # case where no text gets detected at all
        return orig, [], [], [] # <- break out of function very early
        
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    offsets = []
    thetas = []

    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])

    polygons = text_utils.rects2polys(rects, thetas, offsets, rW, rH)
    polygon_pts = []
    rectangle_data = []
    read_polygon_points = False

    # perform non-maxima suppression...
    functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]

    for i, func in enumerate(functions):
        # this implementation uses polygons instead of only rectangles
        # for bounding boxes
        indices = nms.polygons(polygons, confidences, nms_function=func, confidence_threshold=0.5,
            nms_threshold=0.4)

        indices = np.array(indices).reshape(-1)
        polys = np.array(polygons)[indices]

        for poly in polys:
            pts = np.array(poly, np.int32)
            pts = pts.reshape((-1, 1, 2))

            if (show_polys):
                # draw the polygon
                cv.polylines(draw_on, [pts], True, (255, 255, 0), 2)

            if not (read_polygon_points):
                polygon_pts.append(pts)

        read_polygon_points = True

    end = time.time()
    print("Text detection took {:.6f} seconds".format(end - start))
    
    # reset time to track how long text recognition takes,
    # separate from text detection
    start = time.time()

    rois = []
    roi_heights = []
    
    texts_found = []
    text_offsets = []

    counter = 0

    for (startX, startY, endX, endY) in boxes:
        # resize to size of original image
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        # a non-skewed rectangle which entirely contains the rotated
        # rectangle, for skewing purposes
        startX2 = int(min(polygon_pts[counter][0][0][0], polygon_pts[counter][1][0][0],polygon_pts[counter][2][0][0], polygon_pts[counter][3][0][0]))
        endX2 = int(max(polygon_pts[counter][0][0][0], polygon_pts[counter][1][0][0],polygon_pts[counter][2][0][0], polygon_pts[counter][3][0][0]))
        startY2 = int(min(polygon_pts[counter][0][0][1], polygon_pts[counter][1][0][1],polygon_pts[counter][2][0][1], polygon_pts[counter][3][0][1]))
        endY2 = int(max(polygon_pts[counter][0][0][1], polygon_pts[counter][1][0][1],polygon_pts[counter][2][0][1], polygon_pts[counter][3][0][1]))

        # rectangles for each individual text detection
        draw = cv.rectangle(draw_on, \
            (startX2, startY2), (endX2, endY2), (0, 0, 255), 2)

        _w = endX2-startX2
        _h = endY2-startY2

        if mode == "WORD":
            rectangle_data.append((startX2, startY2, endX2, endY2))
        elif mode == "CLUSTER":
            rectangle_data.append((startX2, startY2, _w, _h))
        
        startX3 = int(startX2-((padding/100)*startX2))
        endX3 = int(endX2+((padding/100)*endX2))
        startY3 = int(startY2-((padding/100)*startY2))
        endY3 = int(endY2+((padding/100)*endY2))

        # ensure no bounding box coordinates go outside the bounds
        # of the image; otherwise, script crashes if this happens
        if (startX3 < 0):
            startX3 = 0
        if (endX3 > orig_W-1):
            endX3 = orig_W-1
        if (startY3 < 0):
            startY3 = 0
        if (endY3 > orig_H-1):
            endY3 = orig_H-1        

        # skew correction for heavily rotated images
        roi_reorient = orig[startY3:endY3, startX3:endX3]
        pt0 = polygon_pts[counter][0]
        pt1 = polygon_pts[counter][1]
        text_offsets.append((startX3, startY3))
        if (verbose):
            print ("ROI: {0}".format(roi_reorient.shape))

        # noise removal on extracted ROI done here, if necessary...
        roi_reorient = cv.fastNlMeansDenoisingColored(roi_reorient, None, 10, 10, 7, 21) 

        # scale factor
        f = 0.25
        
        # convert unsigned bytes to int values
        darkest = int(img_color_information.min_max_colors(roi_reorient, f)[0])
        lightest = int(img_color_information.min_max_colors(roi_reorient, f)[1])

        thresh_val = (lightest + darkest) // 2 + 20
        
        roi_reorient = cv.cvtColor(roi_reorient, cv.COLOR_BGR2GRAY)
        roi_reorient = cv.threshold(roi_reorient, thresh_val, 255, cv.THRESH_BINARY)[1]

        # check if image black text on white background, or other way around
        # if more black pixels than white pixels, then invert image
        dom_color = img_color_information.dominant_color(roi_reorient, f)[0]
        if (dom_color == 0):
            roi_reorient = img_color_information.invert_colors(roi_reorient)       

        # figure out size of text we're analyzing
        roi_height = (endY+padding) - (startY-padding)
        if (verbose):
            print ("roi height: {0}".format(roi_height))
        rois.append(roi_reorient)
        roi_heights.append(roi_height)

        # at end of loop, increment counter by 1
        counter = counter + 1

    counter = 0

    if mode == "WORD":
        rect_info = rectangle_data

    # average height of text characters being analyzed
    avg_h = int(statistics.mean(roi_heights))
   
    if mode == "WORD":
        texts = []        
        for roi in rois:
            if (show_rois):
                cv.namedWindow("Final ROI #{0}".format(counter), cv.WINDOW_AUTOSIZE )
                cv.imshow("Final ROI #{0}".format(counter), roi)

            # reads text detections as single words
            tess_config = ("-l eng --oem 1 --psm 8")
            text = pytesseract.image_to_string(roi, config=tess_config)
            texts.append(text)
            print ("Text: {0}".format(text))
            counter = counter + 1

        return texts, rect_info
            
    elif mode == "CLUSTER":
        texts, rect_info = cluster_texts.cluster(orig, rois,\
            rectangle_data, avg_h, rW, rH, padding)

        return texts, rect_info

    # else, attempt to return text detections of single words
    # instead of clusters
    for i in range(0, len(texts)):
        startX = rect_info[i][0]
        startY = rect_info[i][1]
        endX = rect_info[i][2]
        endY = rect_info[i][3]
        draw_on = cv.rectangle(draw_on, (startX, startY), (endX, endY), (0,255,0), 2)
        draw_on = cv.putText(draw_on, texts[i], (rect_info[i][0], rect_info[i][1] - 10), cv.FONT_HERSHEY_SIMPLEX,  
            1, (0,255,0), 2)

# decoding algorithm recycled from Adrian Rosebrock,
# elaborated on by Tom Hoag; article on text detection:
# https://www.pyimagesearch.com/2018/08/20/
# opencv-text-detection-east-text-detector/
def decode(scores, geometry):
    (rows, cols) = scores.shape[2:4]
    rects = []
    confidences = []
    baggages = []

    for y in range(0, rows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, cols):
            if scoresData[x] < min_confidence:
                continue

            # 4.0: offset factor
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            theta = anglesData[x]
            cos = np.cos(theta)
            sin = np.sin(theta)

            # height and width of rectangle
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            offsetX = (endX - startX) // 2
            offsetY = (endY - startY) // 2

            rects.append((startX, startY, endX, endY))

            confidences.append(scoresData[x])
            
            baggages.append({
                "offset": (endX, endY),
                "angle": anglesData[x],
                "xData0": xData0[x],
                "xData1": xData1[x],
                "xData2": xData2[x],
                "xData3": xData3[x]
            })
    
    return (rects, confidences, baggages)
