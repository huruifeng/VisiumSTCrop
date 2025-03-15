import cv2
import numpy as np


def detect_frame_hist(image_file):
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 2)

    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(thresh, (5, 5), 2)

    # 调整HoughCircles参数以提高检测精度
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=0.8,  # 提高dp以检测更明显的圆
        minDist=3,  # 增加点间最小距离避免重叠
        param1=30,
        param2=10,  # 提高param2减少误检
        minRadius=0,
        maxRadius=3
    )

    output = image.copy()
    filtered_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))

        # 提取坐标用于直方图分析
        x_coords = np.array([x for x, y, r in circles])
        y_coords = np.array([y for x, y, r in circles])

        # 确定直方图的边界阈值
        step_size = 2
        threshold = 10  # 每个直方图bin的最小点数

        # x方向计数
        hist_bins = np.arange(np.min(x_coords), np.max(x_coords) + 1, step_size)
        hist_x, _ = np.histogram(x_coords, bins=hist_bins)
        valid_bins_x = np.where(hist_x >= threshold)[0]
        if len(valid_bins_x) > 0:
            x_left = hist_bins[valid_bins_x[0]]
            x_right = hist_bins[valid_bins_x[-1] + 1]
        else:
            x_left, x_right = np.min(x_coords), np.max(x_coords)

        # y方向计数
        hist_bins = np.arange(np.min(y_coords), np.max(y_coords) + 1, step_size)
        hist_y, _ = np.histogram(y_coords, bins=hist_bins)
        valid_bins_y = np.where(hist_y >= threshold)[0]
        if len(valid_bins_y) > 0:
            y_top = hist_bins[valid_bins_y[0]]
            y_bottom = hist_bins[valid_bins_y[-1] + 1]
        else:
            y_top, y_bottom = np.min(y_coords), np.max(y_coords)


        # 过滤位于边界内的点
        filtered_circles = [
            (x, y, r) for x, y, r in circles
            if x >= x_left and x <= x_right and y >= y_top and y <= y_bottom
        ]

        # 绘制检测到的点
        for x, y, r in filtered_circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

        # 可选：绘制边界矩形用于可视化
        cv2.rectangle(output, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (255, 0, 0), 2)

    return filtered_circles, output