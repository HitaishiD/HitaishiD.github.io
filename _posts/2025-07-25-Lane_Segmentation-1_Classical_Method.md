---
layout: post
title: Lane Segmentation with Classical Image Processing Techniques
category: Lane-Segmentation
---

## Introduction

This project focuses on lane detection, a critical task in computer vision which involves identifying the lane area in images or video. Effective lane detection must account for various challenging conditions such as poor lighting, glare, complex road layouts, diverse scenes and varying weather. Lane segmentation plays a vital role in automotive engineering, robotics and intelligent transport systems. The primary motivation behind accomplishing this task is its application in Advanced Driver Assistance Systems (ADAS) and autonomous vehicles. By performing lane detection accurately, these algorithms enhance road safety for humans and increases the reliability of self-driving vehicles.

There exist three main types of image segmentation: semantic segmentation, instance segmentation and panoptic segmentation. This is illustrated in Figure 1. 

1. **Semantic segmentation** involves classifying each pixel in an image to a specific class or category (e.g., car, road, building, etc.) without differentiating between individual instances. All pixels of the same class are assigned the same label.
2. **Instance segmentation** not only labels pixels but also distinguishes between different objects of the same class. In an image with multiple cars, instance segmentation can differentiate each car as a separate entity, even though all are part of the "car" class.
3. **Panoptic segmentation** combines both semantic and instance segmentation. It assigns a semantic label to every pixel, while also distinguishing between different instances of the same class. Thus, panoptic segmentation offers a unified view that captures both the object class and its individual instances.


![Types of Image Segmentation|500](/images/lane-segmentation/segmentation_types.jpg)
<div style="text-align: center;">Figure 1: Types of Image Segmentation</div>

## Dataset 

* **Description of datasets available**
* **State which one is used here and why**
* **Biases in the dataset?**

## Classical Approach

### Overview
We first explore a classical approach using traditional image processing techniques applied sequentially using OpenCV. Traditional image segmentation techniques include k-means clustering and thresholding [1]. However, these techniques come with inherent limitations, for example, the threshold and other parameters must be manually chosen. Therefore, a two-step process was implemented: 
1. Lane boundary detection is performed using a combination of computer vision techniques sequentially applied such as color space conversion, smoothing, edge detection, and Hough transform. This was proposed in [2].
2. Once the lane boundaries are detected, the left and right lines bounding the lane area are joined to form a polygon. This area is filled to represent the detected lane area.

This method does not inherently output a segmentation mask where each pixel is classified into a class. Instead, the above process is used to detect lane boundaries only and subsequently the lane area. 

### Methodology  


The classical method is outlined as follows: 
1. **Load the RGB image:** The RGB image is loaded with OpenCV using its file path.
<div align="center">
  <img src="{{ site.baseurl }}/images/lane-segmentation/0-input-image.png" alt="Input image">
</div>

2. **Color space conversion:** The image is converted from RGB to HSL (Hue, Saturation, Lightness). The HSL color space helps to better isolate colors like yellow and white which are commonly used for lane marking. 
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/1-hsl.png" alt="Input image">
</div>

3. **Masks application:** The range of yellow and white colors is chosen and masks are created for these pixels. These masks are combined and applied to the original image. This helps to filter out irrelevant colors from the image such as road surface, sky, and to focus on the lane area while reducing distractions. It retains spatial information while emphasizing on the lanes.
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/2-mask.png" alt="Input image">
</div>

4. **Grayscale conversion:** Image processing techniques such as edge detection are easier and more effective on grayscale images. This is because it simplifies the calculation of gradients and intensity changes. Therefore, the processing pipeline consists of this step of converting masked images to grayscale.
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/3-grayscale.png" alt="Input image">
</div>

5. **Gaussian blur:** Gaussian blur is a linear filter which helps to smooth the image, by reducing noise and small details that could result in superfluous edges detected further on. This results in the removal of low intensity edges in road images. A Gaussian filter of size 5x5 is used. 
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/4-gaussian-blur.png" alt="Input image">
</div>

6. **Canny edge detection:** This step detects edges by finding regions with significant changes in intensity. It can be used after a 5x5 Gaussian filter has been applied on an image. It requires a high and a low threshold value so that edges with intensity higher than the upper threshold are classified as edges with certainty and those below the lower threshold are discarded. Edges lying between the two thresholds are classified as edges or non-edges based on their connectivity.
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/5-detect-edges.png" alt="Input image">
</div>

7. **Region of Interest (ROI):** The ROI focuses on the area of the image where the lanes most likely appear. This is a step filter and we use an ROI of size (0.5 x height) x width, assuming the camera is mounted on a fixed spot on the car such that the ROI is manually validated. 
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/6-roi.png" alt="Input image">
</div>

8. **Hough transform:** Hough transform is a technique used to detect straight lines in an image, even in noisy environments. Edge detection pre-processing is recommended. This method requires a threshold: it keeps track of the intersection between curves of every point in the image. If the number of intersections is above the threshold, then it declares it as a line with the parameters of the intersection point.
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/7-hough-transform.png" alt="Input image">
</div>

9. **Left and right lane boundaries:** Since several lines are detected from the Hough transform stage, the leftmost and rightmost lines are chosen as the lane boundaries based on their x-coordinates.
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/8-lanes.png" alt="Input image">
</div>

10. **Lane area filling:** A polygon is formed from the left and right lane boundaries. It is colored to represent the lane area.
<div align="center">
	<img src="{{ site.baseurl }}/images/lane-segmentation/9-fill-lane.png" alt="Input image">
</div>

### Results

* **Show example inputs and outputs on the KITTI dataset
* **Discuss limitations**
	* **works in perfect conditions because of the assumptions involved**
	* **no semantic understanding**
	* **points of failure**

<span style="color: rgb(0, 0, 255);">Add results: images and numbers</span>

## From Rule-Based to Learning-Based Lane Detection

To handle more complex scenarios encountered in real-world driving, I turned to semantic segmentation using deep learning based models. 

* First deeplabv3+ model with kitti dataset
* explain how it works
* training setup

## Comparison of Classical and Deep Learning Approaches

* Side-by-side outputs: classical vs deep learning 
* <span style="color: rgb(0, 0, 255);">Add results: images and numbers</span>


## References

[1] W. Chen, W. Wang, K. Wang, Z. Li, H. Li, and S. Liu, “Lane departure warning systems and lane line detection methods based on image processing and semantic segmentation: A review,” Journal of Traffic and Transportation Engineering (English Edition), vol. 7, no. 6, pp. 748–774, Dec. 2020, doi: https://doi.org/10.1016/j.jtte.2020.10.002.

[2] N. Lakhani, R. Karande, and V. Ramakrishnan, “LANE DETECTION USING IMAGE PROCESSING IN PYTHON.” Available: https://www.irjet.net/archives/V9/i4/IRJET-V9I4148.pdf.
‌
‌



