import cv2 as cv
import numpy as np
import math

# invert the colors in this grayscale image
def invert_colors(image, h, w):
    image = cv.bitwise_not(image)
    return image

def reorient_skew(image, index, points):   
    x_coord = 0; y_coord = 1

    draw = image.copy()
    orig = image.copy()

    dilation_kernel = np.ones((5,5), np.uint8) 

    verbose = False
    show_images = False

    # invert colors to white text on black background
    # for contour finding
    (h, w) = image.shape[:2]

    image = invert_colors(image, h, w)

    image_dilated = cv.dilate(image, dilation_kernel, iterations=2)

    # get a contour shaped by the text; image must be white text
    # on black background
    contours, hierarchy = cv.findContours(image_dilated,\
        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    hulls = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hulls.append(hull)

    for i in range(len(contours)):
        cv.drawContours(draw, hulls, i, (0, 0, 0), cv.FILLED)

    # text contour we're interested in is likely the biggest one,
    # if others do end up getting found
    contours = sorted(contours, key=cv.contourArea, reverse=True)
   
    # get each corner point from the contour convex hull
    c = contours[0]

    # tl, tr, br, bl corners of image itself
    img_corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    closest_to_corners = []

    for (corner) in img_corners:
        euclidean_distances = []
        for (pt) in c:
            pt_conv = (pt[0][0], pt[0][1])
            # calculate Euclidean distance using Pythagoras
            dist = math.sqrt(abs(corner[x_coord]-pt_conv[x_coord])**2 +\
                             abs(corner[y_coord]-pt_conv[y_coord])**2)
            euclidean_distances.append((dist, pt_conv))
        closest_to_corners.append(min(euclidean_distances))

    if (show_images):
        cv.namedWindow("Skew{}".format(index), cv.WINDOW_AUTOSIZE );
        cv.imshow("Skew{}".format(index), draw);

    if points is not None:
        tl_point = points[0]
        tr_point = points[1]
        br_point = points[2]
        bl_point = points[3]
    else:
        # top left, top right, bottom right, and bottom_left points
        tl_point = closest_to_corners[0][1]
        tr_point = closest_to_corners[1][1]
        br_point = closest_to_corners[2][1]
        bl_point = closest_to_corners[3][1]

    if (verbose):
        print ("corners: {0}, {1}, {2}, {3}".format(tl_point, tr_point, bl_point, br_point))

    # calculate 'skew' on this rectangular contour
    skew_x = abs(tl_point[x_coord] - bl_point[x_coord])
    skew_y = abs(tl_point[y_coord] - bl_point[y_coord])
    
    if (verbose):
        print ("Skews: ({0}, {1}):".format(skew_x, skew_y))

    p = tl_point
    q = bl_point
    vec_pq = (q[x_coord]-p[x_coord], q[y_coord]-p[y_coord])

    angle = np.arccos(vec_pq[y_coord] / (math.sqrt(vec_pq[x_coord] ** 2 + vec_pq[y_coord] ** 2)))
    # convert angle to degrees
    angle = angle * 180 / math.pi

    if (verbose):
        print ("Angle: {0}".format(angle))

    pad = skew_y // 6 # <- skew_y corresponds to height of
                      #    quadrilateral formed by corner points
                      #    padding at the moment based 1/6 of quad height
        
    # if angle large enough for it to be "worth it" to skew ROI    
    if angle > 15:
        x1 = tl_point[x_coord] - pad; y1 = tl_point[y_coord] - pad
        x2 = tr_point[x_coord] + pad; y2 = tr_point[y_coord] - pad
        x3 = br_point[x_coord] + pad; y3 = br_point[y_coord] + pad
        x4 = bl_point[x_coord] - pad; y4 = bl_point[y_coord] + pad

        # get absolute maximum width of text contour
        widthA = np.sqrt(((br_point[0] - bl_point[0]) ** 2) + ((br_point[1] - bl_point[1]) ** 2))
        widthB = np.sqrt(((tr_point[0] - tl_point[0]) ** 2) + ((tr_point[1] - tl_point[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # + get absolute maximum height of text contour
        heightA = np.sqrt(((tr_point[0] - br_point[0]) ** 2) + ((tr_point[1] - br_point[1]) ** 2))
        heightB = np.sqrt(((tl_point[0] - bl_point[0]) ** 2) + ((tl_point[1] - bl_point[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        src_pts = np.array([
            [x1, y1], [x2, y2],
            [x3, y3], [x4, y4]
            ], dtype="float32")
        dst_pts = np.array([
            [0, 0], [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1], [0, maxHeight-1]
            ], dtype="float32")

        M = cv.getPerspectiveTransform(src_pts,dst_pts)
        skewed = cv.warpPerspective(image, M, (w, h))

        # invert colors back to black text on white background
        skewed = invert_colors(skewed, h, w)

        # convert back to color (opencv BGR format)
        skewed = cv.cvtColor(skewed, cv.COLOR_GRAY2BGR)

        if (verbose):
            print ("New width and height: [{0}, {1}]".format(maxWidth, maxHeight))

        skewed = skewed[0:maxHeight, 0:maxWidth]

        rW = w / maxWidth
        rH = h / maxHeight

        # resize skew corrected image back to size of input image
        skewed = cv.resize(skewed, (int(maxWidth * rW), int(maxHeight * rH)))

        return skewed
    else:
        #get minimum and maximum x,y values from the 4 corners
        min_x = min(tl_point[0], bl_point[0])
        max_x = max(tr_point[0], br_point[0])
        min_y = min(tl_point[1], tr_point[1])
        max_y = max(bl_point[1], br_point[1])

        # then simply get a rectangular ROI...
        image = orig[min_y:max_y,min_x:max_x]
        
        return image

'''
image_path = "images/skewed_text.png"

image = cv.imread(image_path, 1) # 1: color
(H, W) = image.shape[:2]

cv.namedWindow("Before", cv.WINDOW_AUTOSIZE)
cv.imshow("Before", image)

image = reorient_skew(image, 0)

cv.namedWindow("After", cv.WINDOW_AUTOSIZE)
cv.imshow("After", image)

cv.waitKey(0)
'''
