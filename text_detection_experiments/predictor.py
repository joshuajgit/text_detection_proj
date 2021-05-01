import os

import east_w_scores_text_det
from east_w_scores_text_det import EAST_method, east_path
import canny_text_det
from canny_text_det import cho_method
import canny_feature_detection
from canny_feature_detection import canny_feature_extract

#path = "images/Hell.png"
path = "images/Hell_Crop.png"

blobW, blobH = 320, 320
min_conf = 0.95

#east_w_scores_text_det.EAST( \
#    path, east_path, blobW, blobH, min_conf, show=True, counter=count)
#canny_text_det.cho_method( \
#    path, 0, _overlapThresh=0.3, counter=count)
canny_feature_detection.canny_feature_extract( \
    path, blur=4, moment_passes=1, df_tuple=(1, 25), reduce=False, counter=0)
