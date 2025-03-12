import os

import cv2
from matplotlib import pyplot as plt
from sipbuild.generator.outputs import output_code

from DetectFrame_DBSCAN import detect_frame_dbscan
from DetectFrame_HoughLine import detect_frame_houghline
from DetectFrame import detect_frame_dots



image_folder = "Data/images"
image_ls = [img for img in os.listdir(image_folder) if img.endswith(".png")]

output_folder = "outputs"
os.makedirs(output_folder, exist_ok= True)
for i, img in enumerate(image_ls):
    print(i, img)

    original_img = cv2.imread(image_folder + "/" + img)
    circles_dots, output_dots = detect_frame_dots(image_folder + "/" + img)
    circles_dbsacn, output_dbscan = detect_frame_dbscan(image_folder + "/" + img)
    circles_houghline, output_houghline = detect_frame_houghline(image_folder + "/" + img)

    # 显示结果
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 4, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(output_dots)
    plt.title("Detected Dots")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(output_dbscan)
    plt.title("Detected Frame - DBSCAN")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(output_houghline)
    plt.title("Detected Frame - HoughLine")
    plt.axis("off")

    plt.savefig(output_folder +  "/" + img)
    plt.close()