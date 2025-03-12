## VisiumST Dot Frame Detection and Cropping

### Detecting the Dot Frame in VisiumST Images

Use OpenCV for circle detection, such as the `HoughCircles` method, to detect small circular dots on the dot frame. Precisely extract the Visium dot frame using the following steps:

#### Steps:
1. **Preprocessing**: Convert the image to a binary image using adaptive thresholding.
2. **Circle Detection**: Use the `HoughCircles` method to detect small circular dots and obtain possible dot locations (**the number of detected dots is highly sensitive to parameter settings**).
3. **Point Filtering**: Remove dots that do not belong to the dot frame or are located externally.
4. **Bounding Box Calculation**: Ensure that only the dot frame and its internal region are included.

#### Step Explanation:
- Detect all circular dots.
- **Use DBSCAN clustering or Hough transform to identify points on the frame**.
- Compute the **bounding box** of the dot frame.
- **Filter out points outside the bounding box**, keeping only dots inside the frame and removing false detections.
- Display only dots on the dot frame and within its boundaries.

#### DBSCAN Clustering
- Use DBSCAN clustering with an appropriate threshold to group the dots.
- Cluster points and fit straight lines to identify the four edges.
- Filter out points outside the bounding box to ensure only frame points are used.
- Compute the bounding box based on these edge points.

#### Hough Line Detection
- Detect all possible circular dots using `cv2.HoughCircles` to find potential dot frame points in the image.
- Use these detected dots for Hough line transformation, analyze their distribution, and extract the four primary lines (dot frame boundaries).
- Filter out only the points on the dot frame by computing the perpendicular distance from each dot to the detected lines, keeping only dots near the four edges.
- **Note: Some dots may be on the extended line segments but still outside the dot frame**.


------
## VisiumST Dot Frame 检测与裁剪
### 在 VisiumST 图像中检测 dot frame。

使用 OpenCV 进行圆形检测，例如 HoughCircles 来检测 dot frame 上的小圆点。 精准提取 Visium dot frame。
采用以下步骤：
1. **预处理**：使用自适应阈值转换图片为二值图。
2. **圆形检测**：使用 HoughCircles 方法来检测小圆点，得到可能的圆点。（**圆点的数量对参数十分敏感**）。
3. **点筛选**：去除不属于 dot frame或外部的点。
4. **计算外接矩形**：确保仅包含 dot frame 及其内部区域。

#### **步骤说明**
- 先检测所有圆点。
- **使用 DBSCAN 聚类或者霍夫变换找出 frame 上的点**。
- 计算 dot frame 的 **外接矩形**。
- **过滤掉矩形外的点**，仅保留 frame 内部的点，过滤掉误检测到的外部点。
- 只显示 dot frame 上的以及内部的点。

#### **DBSCAN 聚类**
- 使用 DBSCAN 进行点聚类，并选取合适的阈值。
- 先聚类点，并计算直线拟合以找到四条边。
- 过滤掉矩形外部的点，确保计算时仅使用 frame 点。
- 计算这些边界点的外接矩形。

#### **霍夫线 检测**
- 检测所有可能的圆点，使用 cv2.HoughCircles 检测图像中的所有潜在 dot frame 圆点。
- 使用这些圆点进行霍夫线变换,计算点的分布，提取主要的四条线（dot frame 的边界）。
- 筛选出 dot frame 上的点,计算点到直线的垂直距离，只保留靠近四条边的点。
- **注意：有可能点处于线段的延长线上，但是处于 dot frame 外部**


#### **代码实现 (DBSCAN)**

```python
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# 读取图像
image = cv2.imread("visium_image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 预处理：自适应二值化
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# 霍夫圆检测
circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                           param1=50, param2=20, minRadius=3, maxRadius=8)

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
```

