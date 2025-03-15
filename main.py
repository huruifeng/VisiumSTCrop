import os

import cv2
from matplotlib import pyplot as plt

from DetectFrame.DetectDots import detect_frame_dots
from DetectFrame.FilterFrameDots_DBSCAN import filter_dots_dbscan
from DetectFrame.FilterFrameDots_HoughLine import filter_dots_houghline
from DetectFrame.FilterFrameDots_Hist import filter_dots_hist


image_folder = "images"
image_ls = [img for img in os.listdir(image_folder) if img.endswith(".png")]

output_folder = "outputs"
os.makedirs(output_folder, exist_ok= True)
for i, img in enumerate(image_ls):
    print(i, img)

    original_img = cv2.imread(image_folder + "/" + img)
    circles_dots, output_dots = detect_frame_dots(image_folder + "/" + img)
    circles_dbsacn, output_dbscan = filter_dots_dbscan(image_folder + "/" + img)
    circles_houghline, output_houghline = filter_dots_houghline(image_folder + "/" + img)
    circles_hist, output_hist = filter_dots_hist(image_folder + "/" + img)

    # 显示结果
    plt.figure(figsize=(25, 8))

    plt.subplot(1, 5, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(output_dots)
    plt.title("Detected Dots")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(output_dbscan)
    plt.title("Detected Frame - DBSCAN")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(output_houghline)
    plt.title("Detected Frame - HoughLine")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(output_hist)
    plt.title("Detected Frame - Hist")
    plt.axis("off")

    plt.savefig(output_folder +  "/" + img)
    plt.close()