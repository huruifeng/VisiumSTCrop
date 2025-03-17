import os

import cv2
from matplotlib import pyplot as plt

from DetectFrame.DetectDots import detect_frame_dots
from DetectFrame.FilterDots_DBSCAN import filter_dots_dbscan
from DetectFrame.FilterDots_HoughLine import filter_dots_houghline
from DetectFrame.FilterDots_Hist import filter_dots_hist

from DetectFrame.LocateFrame import locate_frame


image_folder = "images"
image_ls = [img for img in os.listdir(image_folder) if img.endswith(".png")]

output_folder = "outputs"
os.makedirs(output_folder, exist_ok= True)
for i, img in enumerate(image_ls):
    print(i, img)

    # 读取图像 Original Image
    original_img = cv2.imread(image_folder + "/" + img)

    # 检测点 Detected Dots
    circles_dots, output_dots = detect_frame_dots(image_folder + "/" + img)

    # 过滤点 Filter Dots
    circles_dbsacn, output_dbscan = filter_dots_dbscan(image_folder + "/" + img)
    circles_houghline, output_houghline = filter_dots_houghline(image_folder + "/" + img)
    circles_hist, output_hist = filter_dots_hist(image_folder + "/" + img)

    # 定位边框 Locate Frame
    circles_frame, output_frame = locate_frame(circles_houghline, image_folder + "/" + img)

    # 显示结果 Show Results
    plt.figure(figsize=(25, 8))

    plt.subplot(1, 6, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.imshow(output_dots)
    plt.title("Detected Dots")
    plt.axis("off")

    plt.subplot(1, 6, 3)
    plt.imshow(output_dbscan)
    plt.title("Detected Frame - DBSCAN")
    plt.axis("off")

    plt.subplot(1, 6, 4)
    plt.imshow(output_houghline)
    plt.title("Detected Frame - HoughLine")
    plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.imshow(output_hist)
    plt.title("Detected Frame - Hist")
    plt.axis("off")

    plt.subplot(1, 6, 6)
    plt.imshow(output_frame)
    plt.title("Detected Frame - Frame")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_folder +  "/" + img)
    plt.close()