import cv2 as cv
import numpy as np
import time
from imutils.object_detection import non_max_suppression

import pytesseract

east_path = "learning_models/text_processing/frozen_east_text_detection.pb"

# default values for east blob size
east_w = 320
east_h = 320

# confidence level required to be met to say a detected region is text
min_confidence = 0.99

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

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

# decoding algorithm recycled from Adrian Rosebrock,
# elaborated on by Tom Hoag
# article on text detection:
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

#path = "images/starbucks_color.png"
path = "images/KAIST/DSC02629.jpg"

# track time to test algorithm's speed...
def EAST_method(image_path, east_path, blobW, blobH, min_confidence, show, counter):
    image = cv.imread(image_path, 1)
    draw, orig = image.copy(), image.copy()
    
    start = time.time()

    (H, W) = image.shape[:2]
    (orig_H, orig_W) = image.shape[:2]

    print ("EAST dims: {0}x{1}".format(east_w, east_h))

    newW, newH = (east_w, east_h)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # load model
    net = cv.dnn.readNet(east_path)

    blob = cv.dnn.blobFromImage(image, 1.0, (newW, newH),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)
    (rects, confidences, baggage) = decode(scores, geometry)

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    boxes = np.sort(boxes, axis=0)

    all_texts = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv.rectangle(draw, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
        counter += 1

        if startX < 0:
            startX = 0
        if startY < 0:
            startY = 0
        if startX > orig_W:
            startX = orig_W-1
        if startY > orig_H:
            startY = orig_H-1       

        '''            
        roi = orig[startY:endY, startX:endX]

        texts_found = tesseract_func(roi, 0)
        all_texts.append(texts_found[0][1])
        '''
    
    #if show:
    #    cv.imshow('img_EAST{0}'.format(counter+1), draw)

    cv.imwrite('images/results/east/{0}.jpg'.format(counter), draw)

    '''
    text_final = " ".join(all_texts)
    print (text_final)

    return text_final
    '''
#EAST_method(path, east_path, east_w, east_h, min_confidence, show=True, counter=0)
