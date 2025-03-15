import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def filter_frame_dots(circles, image_file=None, method="DBSCAN"):
    if image_file is not None:
        image = cv2.imread(image_file)

    filtered_circles = []

    circles = np.uint16(np.around(circles[0, :]))

    # 仅保留点坐标
    circle_points = np.array([[x, y] for x, y, r in circles])

    if method == "DBSCAN":
        # 使用 DBSCAN 聚类点，找到四条边上的点（frame 上的点应成线状）
        clustering = DBSCAN(eps=20, min_samples=5).fit(circle_points)

        # 统计哪些簇可能是 dot frame
        labels = clustering.labels_
        unique_labels = set(labels)

        # 只保留可能是 frame 的簇（去除 -1 噪声簇）
        frame_points = circle_points[labels != -1]

        # 计算 dot frame 的外接矩形（四周只包含 frame 及其内部区域）
        x_min, y_min = np.min(frame_points, axis=0)
        x_max, y_max = np.max(frame_points, axis=0)

        # 过滤掉矩形外的点
        filtered_circles = [p for p in circles if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max]

        if image_file is not None:
            for x, y, r in filtered_circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # 绿色圆圈
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # 红色中心点

    elif method == "HoughLine":
        # 创建一个黑色背景，用于存放圆点
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dot_mask = np.zeros_like(gray)

        # 画出所有检测到的圆点
        for (x, y) in circle_points:
            cv2.circle(dot_mask, (x, y), 2, (255, 255, 255), -1)

        # 进行边缘检测
        edges = cv2.Canny(dot_mask, 50, 150, apertureSize=3)

        # 霍夫线变换
        """
        参数调整建议
        1. rho 和 theta
            rho=1，theta=np.pi/180 适用于大多数情况（像素单位和 1° 分辨率）
            如果检测结果太密集，可以增大 rho 或 theta 以减少噪声。
        2. threshold
            值大（如 100） → 只检测最明显的直线（减少误检）。
            值小（如 20） → 也会检测较弱的直线（可能有噪声）。
        3. minLineLength
            这个值控制最短要检测的直线长度，防止碎片化。
            例如：在 Visium dot frame 上，你可以调大（如 100），避免检测短的噪声线段。
        4. maxLineGap
            允许断续的线段连接，适用于点阵结构。
            如果线条断裂但应该连在一起，可以适当增大，比如 20。
        """
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=10, maxLineGap=20)

        # 存储检测到的直线
        frame_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                frame_lines.append([(x1, y1), (x2, y2)])
                if image_file is not None:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色线条

        # **寻找四边形框架**
        frame_points = np.array(circle_points)
        x_min, y_min = np.min(frame_points, axis=0)
        x_max, y_max = np.max(frame_points, axis=0)

        # **第三步：筛选出真正属于 frame 的点**

        for (x, y) in circle_points:
            # 计算点到所有霍夫线的最小距离
            min_dist = float("inf")
            closest_segment = None

            for (x1, y1), (x2, y2) in frame_lines:
                # 计算点到直线的垂直距离
                distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

                # 记录最近的线段
                if distance < min_dist:
                    min_dist = distance
                    closest_segment = [(x1, y1), (x2, y2)]

            # 约束 1：点必须在外接矩形内（避免远离 frame 的点被误识别）
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue  # 直接跳过这个点

            # 约束 2：点可以在线段上，也可以稍微超出线段（frame 上）
            if closest_segment:
                x1, y1 = closest_segment[0]
                x2, y2 = closest_segment[1]

                # 计算点在线段的投影
                dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
                segment_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

                # 允许点在线段的外部延长线上 **但误差要小**
                if -30 <= dot_product <= segment_length_sq + 30:  # 允许稍微超出
                    filtered_circles.append((x, y, 3))
                    if image_file is not None:
                        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # 红色点表示保留的点
    else:
        # 提取坐标用于直方图分析
        x_coords = np.array([x for x, y, r in circles])
        y_coords = np.array([y for x, y, r in circles])

        # 确定直方图的边界阈值
        step_size = 5
        threshold = 5  # 每个直方图bin的最小点数

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

        if image_file is not None:
            # 绘制检测到的点
            for x, y, r in filtered_circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    if image_file is not None:
        return filtered_circles, image
    return filtered_circles

