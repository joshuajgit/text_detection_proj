# executable script
import sys
sys.path.append('_text_processing')

import cv2 as cv
import numpy as np
import text_recognition
import text_recognition_canny

x_coord = 0; y_coord = 1

image_path = "images/guardian_unity.png"
#image_path = "images/KAIST/080119-0014.jpg"
#image_path = "images/KAIST/080119-0009.jpg"
#image_path = "images/KAIST/080119-0038.jpg"
#image_path = "images/wine.png"
#image_path = "images/wine2x.png"
#image_path = "images/wine4x.png"
#image_path = "images/clusters_testing.png"
#image_path = "images/KAIST/DSC02925.jpg"
image_path = "images/starbucks_color.png"
#image_path = "images/KAIST/!!080116-0053.jpg"
#image_path = "images/KAIST/DSC02587.jpg"
#image_path = "images/bolt.png"

image_paths_multi = [
    "images/guardian_unity.png",
    "images/KAIST/080119-0014.jpg",
    "images/KAIST/080119-0009.jpg",
    "images/KAIST/080119-0038.jpg",
    "images/wine4x.png",
    "images/clusters_testing.png",
    "images/KAIST/DSC02925.jpg",
    "images/starbucks_color.png",
]

image = cv.imread(image_path, 1) # 1: color
draw = image.copy(); orig = image.copy()

(h, w) = image.shape[:2]

input_ = "SINGLE" # <- single or multiple

mode = "WORD" # <- character, word or cluster
algorithm = "CANNY" # <- east or canny
edge_type = "AUTO" # <- auto or tight

net = cv.dnn.readNet(text_recognition.east_path)

mode = mode.upper()
algorithm = algorithm.upper()
edge_type = edge_type.upper()

if algorithm == "EAST":
    texts, rect_info = \
        text_recognition.text_recognition(image, net, mode)
    
    # drawing part occurs once text location data returned
    counter = 0
    for rect in rect_info:
        if (texts[counter] != ""):          
            startX = rect[0]
            startY = rect[1]
            endX = rect[2]
            endY = rect[3]
            image = cv.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
            image = cv.putText(image, texts[counter], (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX,  
                1, (0,255,0), 2)

        counter += 1

    cv.namedWindow("Final result", cv.WINDOW_AUTOSIZE )
    cv.imshow("Final result", image)
    
elif algorithm == "CANNY":
    if input_ == "SINGLE":
        texts, poly_info = \
            text_recognition_canny.text_recognition_canny(image, mode, edge_type)

        # drawing part occurs once text location data returned
        counter = 0
        for poly in poly_info:
            if (texts[counter] != ""):            
                tl_point, tr_point, bl_point, br_point = \
                    poly[0], poly[1], poly[2], poly[3]

                image = cv.line(image, (tl_point), (tr_point), (0, 255, 0), 2)
                image = cv.line(image, (tr_point), (br_point), (0, 255, 0), 2)
                image = cv.line(image, (br_point), (bl_point), (0, 255, 0), 2)
                image = cv.line(image, (bl_point), (tl_point), (0, 255, 0), 2)

                image = cv.putText(image, texts[counter], \
                    (tl_point[0], tl_point[1] - 10), \
                    cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            counter += 1

        cv.namedWindow("Final result", cv.WINDOW_AUTOSIZE )
        cv.imshow("Final result", image)

    elif input_ == "MULTIPLE":
        image_count = 0
        for image_path in image_paths_multi:
            image = cv.imread(image_path, 1) # 1: color
            
            texts, poly_info = \
                text_recognition_canny.text_recognition_canny(image, mode, edge_type)

        # drawing part occurs once text location data returned
        counter = 0
        for poly in poly_info:
            if (texts[counter] != ""):            
                tl_point, tr_point, bl_point, br_point = \
                    poly[0], poly[1], poly[2], poly[3]

                image = cv.line(image, (tl_point), (tr_point), (0, 255, 0), 2)
                image = cv.line(image, (tr_point), (br_point), (0, 255, 0), 2)
                image = cv.line(image, (br_point), (bl_point), (0, 255, 0), 2)
                image = cv.line(image, (bl_point), (tl_point), (0, 255, 0), 2)

                image = cv.putText(image, texts[counter], \
                    (tl_point[0], tl_point[1] - 10), \
                    cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            counter += 1

            cv.namedWindow("Final result #{0}".format(image_count), cv.WINDOW_AUTOSIZE )
            cv.imshow("Final result #{0}".format(image_count), image)

        image_count += 1

cv.waitKey(0)
