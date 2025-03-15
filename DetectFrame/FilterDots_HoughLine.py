import os
import cv2
import numpy as np


def filter_dots_houghline(image_file):
    image = cv2.imread(image_file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 81, 2)

    # 高斯模糊，减少噪声
    """
    增加 sigmaX 可以增加模糊程度，但可能会增加噪声
    增加 ksize 可能增加检测到的圆的数量，可能会导致误检测
    """
    blurred = cv2.GaussianBlur(thresh, (3, 3), 2)

    # Detect circles using HoughCircles
    """
    参数调整建议
    如果 检测不到圆：
        降低 param2（如 30 -> 20）。
        增加 minDist，防止过多重复检测。
    如果 检测到了很多误检：
        增加 param2（如 30 -> 50）。
        限制 minRadius 和 maxRadius 以避免非目标圆。
    
    检测不到 dot frame：
        增大 dp（如 1.2 -> 1.5）
    检测到了很多外部圆：
        减小 dp（如 1.2 -> 0.6）
    """
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.6, minDist=3,param1=50, param2=10, minRadius=0, maxRadius=3)

    # 确保检测到圆点
    all_circle_points = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        all_circle_points = [(x, y) for x, y, r in circles]

    # 创建输出图像
    output = image.copy()
    filtered_points = []

    # 第二步：使用所有检测到的圆点来进行霍夫线变换
    if all_circle_points:
        # 创建一个黑色背景，用于存放圆点
        dot_mask = np.zeros_like(gray)

        # 画出所有检测到的圆点
        for (x, y) in all_circle_points:
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
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=10, maxLineGap=20)

        # 存储检测到的直线
        frame_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                frame_lines.append([(x1, y1), (x2, y2)])
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色线条

        # **寻找四边形框架**
        frame_points = np.array(all_circle_points)
        x_min, y_min = np.min(frame_points, axis=0)
        x_max, y_max = np.max(frame_points, axis=0)

        # **第三步：筛选出真正属于 frame 的点**

        for (x, y) in all_circle_points:
            # 计算点到所有霍夫线的最小距离
            min_dist = float("inf")
            closest_segment = None

            for (x1, y1), (x2, y2) in frame_lines:
                # 计算点到直线的垂直距离
                distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / \
                           np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

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
                    filtered_points.append((x, y))
                    cv2.circle(output, (x, y), 3, (0, 0, 255), -1)  # 红色点表示保留的点

    return filtered_points, output

    # # 显示最终结果
    # cv2.imshow("Filtered Dot Frame", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
