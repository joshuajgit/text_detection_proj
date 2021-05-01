import math
import numpy as np
import cv2
import imutils

import pytesseract

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

def get_canny(image, edge_type):
    # apply automatic Canny edge detection using the computed median
    # thanks to Adrian Rosebrock

    if edge_type == "AUTO":
        # AUTO
        v = np.median(image)
        sigma = 0.33
        
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        print ("Canny: <{0}, {1}>".format(lower, upper))

        canny = cv2.Canny(image, lower, upper)

    if edge_type == "TIGHT":
        # TIGHT EDGES
        # using blurred image to try to reduce edges found to most likely
        # candidates for text

        canny = cv2.Canny(image, 200, 250)

    return canny

path = "images/KAIST/DSC02629.jpg"

def canny_feature_extract(image_path, blur, moment_passes, df_tuple, reduce, counter):
    image = cv2.imread(image_path, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    draw, show_text = image.copy(), image.copy()

    if blur > 0:
        kernel = np.ones((blur,blur),np.float32)/(blur*blur)
        gray = cv2.filter2D(gray, -1, kernel)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    gray = cv2.Canny(gray, 50, 100)

    dilation_kernel = np.ones((df_tuple[0],df_tuple[1]), np.uint8)
    gray = cv2.dilate(gray, dilation_kernel, iterations=2)

    #cv2.imshow('CANNY{0}'.format(counter), gray)

    cnts = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:1]
    contour = cnts[0]

    for i in range(moment_passes):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.drawContours(draw, [contour], -1, (0, 255, 0), 3)
        cv2.circle(draw, (cX, cY), 7, (255, 255, 255), -1)

        subset = None
        # remove 1/4 of the contour edges
        if reduce:
            subset = int(len(contour) / 4)

        x, y = 0, 1
        distances = []
        reduced_contour = []
        for i in range(0, len(contour)):
            dist = math.sqrt(abs(contour[i][0][x]-cX)**2 + \
                             abs(contour[i][0][y]-cY)**2)
            distances.append((dist, i))

        distances = sorted(distances, reverse = True)

        if reduce:
            indices = []
            for i in range(0, subset):
                indices.append(distances[i][1])

            new_contour = np.zeros([(len(contour)-subset), 1, 2])
            new_contour = new_contour.astype(int)

            old_ind, new_ind = 0, 0
            for coordinate in contour:
                if old_ind not in indices:
                    new_contour[new_ind] = coordinate
                    new_ind += 1
                old_ind += 1
        else:
            new_contour = contour

        cv2.drawContours(draw, [new_contour], -1, (0, 255, 0), 2)
        
        contour = new_contour

    blank = np.zeros((image.shape[0],image.shape[1],3), np.uint8)

    cv2.drawContours(blank, [new_contour], -1, (255, 255, 255), 3)
    cv2.fillPoly(blank, pts=[new_contour], color=(255, 255, 255))

    erosion_kernel = np.ones((10,10), np.uint8)
    blank = cv2.erode(blank, erosion_kernel, iterations=2)

    x,y,w,h = cv2.boundingRect(new_contour)

    roi = image[y:y+h,x:x+w]

    #cv2.imshow('Canny edges', gray)
    #cv2.imshow('roi{0}'.format(counter), roi)
    
    #cv2.imshow('img_feat_det{0}'.format(counter), draw)

    cv2.imwrite('images/results/canny/{0}.jpg'.format(counter), draw)
    #cv2.imshow('BLANK{0}'.format(counter), blank)

    '''
    texts_found = tesseract_func(roi, 0)

    if len(texts_found) > 0:        
        cv2.rectangle(show_text, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(show_text, "{}".format(texts_found[0][1]), (x, y-10), \
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        return texts_found[0][1]
    else:
        return ""
    '''
    
#print (canny_feature_extract(path, blur=2, moment_passes=1, df_tuple=(1, 27), reduce=False, counter=0))


