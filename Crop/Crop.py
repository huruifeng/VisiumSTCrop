import cv2
import numpy as np

from DetectFrame.FilterDots_DBSCAN import detect_frame_dbscan



def crop(image_file):
    circles_dbsacn, output_dbscan = detect_frame_dbscan(image_file)

    x_coords = np.array([x for x, y, r in circles_dbsacn])
    y_coords = np.array([y for x, y, r in circles_dbsacn])

    x_left = np.min(x_coords)
    x_right = np.max(x_coords)
    y_top = np.min(y_coords)
    y_bottom = np.max(y_coords)

    return x_left, x_right, y_top, y_bottom