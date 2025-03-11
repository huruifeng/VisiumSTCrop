import os
import cv2
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

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


# 如果检测到圆，则进行绘制
if circles is not None:
    circles = np.uint16(np.around(circles))  # 将坐标转换为整数
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画圆
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), -1)  # 画圆心

# 显示结果
cv2.imshow("Detected Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
