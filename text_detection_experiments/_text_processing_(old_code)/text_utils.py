# Used in Tom Hoag's implementation of EAST text/region detection
# https://bitbucket.org/tomhoag/opencv-text-detection/src/master/opencv_text_detection/

import math

def rects2polys(rects, thetas, origins, rW, rH):
    polygons = []
    for i, box in enumerate(rects):
        upperLeftX = box[0]
        upperLeftY = box[1]
        lowerRightX = box[2]
        lowerRightY = box[3]

        upperLeftX = int(upperLeftX * rW)
        upperLeftY = int(upperLeftY * rH)
        lowerRightX = int(lowerRightX * rW)
        lowerRightY = int(lowerRightY * rH)

        # rectangle vertices
        points = [
            (upperLeftX, upperLeftY),
            (lowerRightX, upperLeftY),
            (lowerRightX, lowerRightY),
            (upperLeftX, lowerRightY)
        ]

        rotationPoint = (int(origins[i][0] * rW), int(origins[i][1] * rH))

        rotatedPoints = rotatePoints(points, thetas[i], rotationPoint)

        polygons.append(rotatedPoints)

    return polygons

def rotatePoints(points, theta, origin):
    rotated = []
    for xy in points:
        rotated.append(rotate_around_point(xy, theta, origin))

    return rotated

def rotate_around_point(xy, radians, origin):    
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy
