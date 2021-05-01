import cv2 as cv
import numpy as np
import math

from nms import nms
from nms import malisiewicz

import sys
sys.path.append('_text_processing')

import img_color_information
import skew_correction
import canny

import pytesseract

# --- Hyperparameters ---
# have these add up to 1, experiment with these for best combination
likelihood_colors = 0.20
likelihood_distance = 0.48
likelihood_size = 0.20
likelihood_ar = 0.14

thresh = 0.95
# -----------------------

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# the calls to Tesseract that each thread makes are here
def tesseract_func(roi, counter):
    texts_found = []
    
    config = ("-l eng --oem 1 --psm 6")
    #config = ("-l eng --oem 1 --psm 8")
    text = pytesseract.image_to_string(roi, config=config)
    print ("TEXT #{0}: {1}\n".format(counter, text))

    # remove any non-ASCII characters that may be picked
    # up by mistake, and any newline characters
    text = ''.join(i for i in text if ord(i) < 128).strip()
    text = text.replace('\n', ' ')

    texts_found.append((counter, text))

    return texts_found

# faster NMS method implemented by Adrian Rosebrock:
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
	
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        largest_x1 = np.maximum(x1[i], x1[idxs[:last]])
        largest_y1 = np.maximum(y1[i], y1[idxs[:last]])
        largest_x2 = np.minimum(x2[i], x2[idxs[:last]])
        largest_y2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, largest_x2 - largest_x1 + 1)
        h = np.maximum(0, largest_y2 - largest_y1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def misc_suppression(boxes, _type, arg):
    AR_THRESH = 8
    SMALL_SIZE_THRESH = 20
    pick = []

    i = 0
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        ar = w / h if w > h else h / w

        if _type == "aspectratio":
            if ar < AR_THRESH:
                pick.append(boxes[i])
        elif _type == "small":
            if w > SMALL_SIZE_THRESH or h > SMALL_SIZE_THRESH:
                pick.append(boxes[i])
        elif _type == "large":
            LARGE_SIZE_THRESH = (w * h) * 2.5
            if LARGE_SIZE_THRESH < arg:
                pick.append(boxes[i])
        elif _type == "area":
            if w * h < arg * 8:
                pick.append(boxes[i])

        i += 1

    return pick

# aspect ratio of ROI
def get_aspect_ratio(w, h):
    # will be > 1 with bigger width,
    # < 1 with bigger height
    return w / h

path = "images/KAIST/DSC02629.jpg"

_overlapThresh = 0.3

def cho_method(image_path, blur, _overlapThresh, counter):
    mser = cv.MSER_create()

    image = cv.imread(image_path, 1)

    if blur > 0:
        kernel = np.ones((blur,blur),np.float32)/(blur*blur)
        image = cv.filter2D(image,-1,kernel)

    imgH, imgW = image.shape[:2]

    orig = image.copy()
    vis1, vis2, vis3, vis4 = image.copy(), image.copy(), image.copy(), image.copy()

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    regions, _ = mser.detectRegions(gray)

    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    rects = []
    for i in range(len(hulls)):
        x,y,w,h = cv.boundingRect(hulls[i])
        rects.append((x,y,x+w,y+h))

    polygons = hulls

    # NMS
    rects = misc_suppression(rects, "large", imgH * imgW)
    areas = []

    for i in range(len(rects)):
        x1,y1,x2,y2 = rects[i]
        cv.rectangle(vis1,(x1,y1),(x2,y2),(255,0,0),1)
        areas.append((x2-x1)*(y2-y1))

    areas.sort()
    cut = int(len(areas)/20)
    mean = sum(areas) / len(areas)
    median = areas[int(len(areas)/2)]
    #print ("mean:", mean); print ("median:", median)

    # pre NMS candidate filtering
    rects = misc_suppression(rects, "area", median)
    boxes = np.array(rects, dtype=np.float32)
    boxes = non_max_suppression(boxes, _overlapThresh)

    # post NMS candidate filtering
    #boxes = misc_suppression(boxes, "aspectratio", None)
    boxes = misc_suppression(boxes, "small", None)

    rois = []
    sizes = []
    spatial_locations = []
    ycrcb_colors = []
    aspect_ratios = []

    for i in range(len(boxes)):
        x1,y1,x2,y2 = boxes[i]
        cv.rectangle(vis2,(x1,y1),(x2,y2),(0,0,255),1)
        cv.putText(vis2, "{}".format(i), (x1, y1-10), \
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        roi = orig[y1:y2, x1:x2]
        roi_ycrcb = cv.cvtColor(roi, cv.COLOR_BGR2YCrCb)

        channels_ycrcb = cv.mean(roi_ycrcb)

        w, h = x2-x1, y2-y1
        sizes.append((w, h))
        spatial_locations.append((x1, y1))
        ycrcb_colors.append(channels_ycrcb)
        aspect_ratios.append(get_aspect_ratio(w, h))

    # coords for most likely regions of text
    best_candidates = []
    region_connections = []

    verbose = False

    X_, Y_ = 0, 1
    # O(n^2) complexity. This won't scale very well.
    for i in range(len(boxes)):
        if verbose:
            print ("comparing all other values to that in index {0} ({1})".format(i, spatial_locations[i]))
        for j in range(len(boxes)):
            #if i != j and i < j:
            if i != j:
                # 4.4. based off of Text Tracking by Hysteresis section in Cho paper
                likelihood = 0
                
                # compare distances
                dist = math.sqrt((spatial_locations[j][X_]-spatial_locations[i][X_])**2+ \
                       (spatial_locations[j][Y_]-spatial_locations[i][Y_])**2)
                if (dist < 2 * max(sizes[i][X_], sizes[i][Y_])):
                    if verbose:
                        print ("{} -> {}; close".format(j, spatial_locations[j]))
                    likelihood += likelihood_distance
                else:
                    if verbose:
                        print ("{} -> {}; far".format(j, spatial_locations[j]))
                
                # compare sizes
                if (abs(sizes[j][Y_] - sizes[i][Y_]) < min(sizes[i][Y_], sizes[j][Y_])):
                    if verbose:
                        print ("{} -> {}; similar size".format(j, sizes[j]))
                    likelihood += likelihood_size
                else:
                    if verbose:
                        print ("{} -> {}; dissimilar size".format(j, sizes[j]))

                # compare color spaces
                similar_colors = True
                for k in range(3): # num channels
                    if (abs(ycrcb_colors[j][k] - ycrcb_colors[i][k]) >= 25):
                        similar_colors = False

                if similar_colors:
                    if verbose:                
                        print ("{}; similar colors".format(j))
                    likelihood += likelihood_colors
                else:
                    if verbose:                
                        print ("{}; dissimilar colors".format(j))

                # compare aspect ratios
                ar_i, ar_j = aspect_ratios[i], aspect_ratios[j]
                ar = ar_i / ar_j if ar_i > ar_j else ar_j / ar_i
                if abs(ar_j - ar_i) <= 0.5:
                    if verbose:                
                        print ("{} -> {}; similar aspect ratio".format(j, aspect_ratios[j]))
                    likelihood += likelihood_ar
                else:
                    if verbose:                
                        print ("{} -> {}; dissimilar aspect ratio".format(j, aspect_ratios[j]))

                if likelihood > thresh:
                    if verbose:
                        print ("i, j: {}, {}".format(i, j))
                    region_connections.append((i, j))

    # todo: make the clusters; start by checking all connections
    '''
    clusters = []
    '''

    best_candidates = []

    for i, j in region_connections:
        if i not in best_candidates:
            best_candidates.append(i)
        if j not in best_candidates:
            best_candidates.append(j)

    boxes_final = []
    for i in range(len(best_candidates)):
        boxes_final.append(boxes[best_candidates[i]])

    x1s, y1s, x2s, y2s = [], [], [], []

    for i in range(len(boxes_final)):
        x1,y1,x2,y2 = boxes_final[i]
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)    
        cv.rectangle(vis3,(x1,y1),(x2,y2),(0,255,255),1)
        cv.putText(vis3, "{}".format(i), (x1, y1-10), \
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)

    if len(boxes_final) > 0:
        padding = 0
        bound_rect = (min(x1s)-padding, min(y1s)-padding, \
                      max(x2s)+padding, max(y2s)+padding)

        cv.rectangle(vis3,(bound_rect[0],bound_rect[1]),
                          (bound_rect[2],bound_rect[3]),(0,255,0),2)
        cv.rectangle(vis4,(bound_rect[0],bound_rect[1]),
                          (bound_rect[2],bound_rect[3]),(0,255,0),2)

        #cv.imshow('img_cho_det{0}'.format(counter), vis3)

        cv.imwrite('images/results/mser/{0}.jpg'.format(counter), vis3)

        '''
        roi = orig[bound_rect[1]:bound_rect[3], bound_rect[0]:bound_rect[2]]

        text_found = tesseract_func(roi, 0)

        cv.putText(vis4, "{}".format(text_found[0][1]), (bound_rect[0], bound_rect[1]-10), \
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        #cv.imshow('text', vis4)

        return text_found[0][1]
        '''
    #else:
    #    return ""

#print (cho_method(path, 0, _overlapThresh=0.3, counter=0))
