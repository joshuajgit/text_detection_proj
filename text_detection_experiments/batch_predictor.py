import os

import east_w_scores_text_det
from east_w_scores_text_det import EAST_method, east_path
import canny_text_det
from canny_text_det import cho_method
import canny_feature_detection
from canny_feature_detection import canny_feature_extract

from difflib import SequenceMatcher

def assess_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

img_folder = "images/KAIST"

files = []
for f in os.listdir(img_folder):
    files.append("{0}/{1}".format(img_folder, f))

blobW, blobH = 320, 320
min_conf = 0.95

east_detections = []
cho_based_detections = []
feature_based_detections = []

examples = 50

count = 0
for f in files:
    if count < examples:
        det1 = east_w_scores_text_det.EAST_method( \
            f, east_path, blobW, blobH, min_conf, show=False, counter=count)
        det2 = canny_text_det.cho_method( \
            f, 0, _overlapThresh=0.3, counter=count)
        det3 = canny_feature_detection.canny_feature_extract( \
            f, blur=2, moment_passes=1, df_tuple=(1, 25), reduce=False, counter=count)

        east_detections.append(det1)
        cho_based_detections.append(det2)
        feature_based_detections.append(det3)
        
    elif count > examples:
        break
    count += 1

'''
print ("EAST detections:",east_detections)
print ("Cho based detections:",cho_based_detections)
print ("Feature based detections:",feature_based_detections)

correct_preds = []
doc = open("images/preds.txt", "r")
text = doc.read()
for line in text.split("\n"):
    correct_preds.append(line)

print (correct_preds[0:examples])

print ("EAST detections similarities:")
for i in range(examples):
    print (assess_similarity(east_detections[i].lower(), correct_preds[i]))
print ("Cho based detections similarities:")
for i in range(examples):
    print (assess_similarity(cho_based_detections[i].lower(), correct_preds[i]))
print ("Feature based detections similarities:")
for i in range(examples):
    print (assess_similarity(feature_based_detections[i].lower(), correct_preds[i]))

doc.close()
'''
