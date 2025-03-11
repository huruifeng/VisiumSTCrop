import os
import cv2
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

image = cv2.imread(os.path.join("images", "HC_BN1762_MTG_VisiumST_batch25_rep1.png"))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 81, 2)

# 高斯模糊，减少噪声
"""
增加 sigmaX 可以增加模糊程度，但可能会增加噪声
增加 ksize 可能增加检测到的圆的数量，可能会导致误检测
"""
blurred = cv2.GaussianBlur(thresh, (5, 5), 2)

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
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.6, minDist=3,param1=30, param2=10, minRadius=0, maxRadius=3)

# 创建显示结果的副本
output = image.copy()

if circles is not None:
    circles = np.uint16(np.around(circles[0, :]))

    # 仅保留点坐标
    circle_points = np.array([[x, y] for x, y, r in circles])

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

    # 绘制检测到的 dot frame 圆点
    for x, y, r in filtered_circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # 绿色圆圈
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)  # 红色中心点

# 显示结果
cv2.imshow("Filtered Dot Frame Circles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
