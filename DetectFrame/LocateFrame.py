import cv2
import numpy as np


def locate_frame(circles, image_file=None):
    if image_file is None:
        print("Error: image_file or coordinates is None")
        return None

    image = cv2.imread(image_file)
    height, width, _ = image.shape

    x_coords = np.array([x for x, y, r in circles])
    y_coords = np.array([y for x, y, r in circles])

    # 确定直方图的边界阈值
    step_size = 2
    threshold = 5  # 每个直方图bin的最小点数

    valid_x = []
    for i in range(0, height, step_size):
        count_x = len([x for x in x_coords if x >= i and x < i + step_size])
        if count_x >= threshold:
            valid_x.append(i)

    valid_y = []
    for i in range(0, width, step_size):
        count_y = len([y for y in y_coords if y >= i and y < i + step_size])
        if count_y >= threshold:
            valid_y.append(i)

    valid_circles = [p for p in circles if p[0] in valid_x or p[1] in valid_y] # 使用OR, 不使用AND

    x_left = np.min(valid_circles, axis=0)[0]
    x_right = np.max(valid_circles, axis=0)[0]
    y_top = np.min(valid_circles, axis=0)[1]
    y_bottom = np.max(valid_circles, axis=0)[1]


    for x, y, r in valid_circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

    cv2.rectangle(image, (x_left, y_top), (x_right, y_bottom), (0, 0, 255), 2)


    return valid_circles, image