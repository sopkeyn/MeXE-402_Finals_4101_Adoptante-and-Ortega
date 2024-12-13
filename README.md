<div align="center">

# 🎯 **MeXE 402 Finals - 4101: Adoptante and Ortega**

</div>

![Orange Brown Cute Illustrative Bingo Game Presentation (6)](https://github.com/user-attachments/assets/34c913b1-77ba-4a5a-8cfd-edb857ca5351)
![Orange Brown Cute Illustrative Bingo Game Presentation (1)](https://github.com/user-attachments/assets/79757aac-23b3-467d-8eae-b7f4ad8cd286)
# 🖼️ Outlining and Highlighting Road Signs Images Using Contour Detection 

## 🚦 Road Signs

### 📖 Introduction
Welcome to the **Road Signs** repository! This project explores the critical task of traffic sign recognition using computer vision techniques. Our focus is on leveraging contour detection to accurately identify and interpret road signs, ensuring safer navigation for drivers and autonomous vehicles.

This repository focuses on commonly recognized traffic signs, including:
- 🛑 **Speed Signs**
- ⚠️ **Yield Signs**
- 🚸 **Pedestrian Crossing Signs**
- 🅿️ **Parking Signs**

---

### ❓ Problem Statement and Significance

🚗 **Traffic Sign Recognition**  
Traffic sign recognition is a crucial application in computer vision, directly impacting:
- 🛡️ **Road Signs**: Ensuring drivers receive clear information about road conditions and regulations.
- 🤖 **Autonomous Vehicles**: Enabling self-driving systems to navigate efficiently and safely.  
Accurate traffic sign recognition reduces accidents and enhances traffic management.

🔺 **Key Characteristics of Road Signs**  
Road signs are designed with simple, easily distinguishable shapes like:
- 🔴 **Circles**: Example: Speed signs.
- 🔼 **Triangles**: Example: Yield signs.
- 🔷 **Rectangles**: Example: Information signs.

These shapes make road signs ideal candidates for computer vision techniques like contour detection.

---

### 🧪 Relevance to Contour Detection
Contour detection is a foundational method in computer vision used to:
- 🖼️ Identify object boundaries in images.
- 🔍 Recognize geometric shapes and patterns.

This technique aligns perfectly with detecting road signs by outlining their contours, such as:
- A **circle** for speed signs 🛑.
- A **triangle** for yield signs ⚠️.
- A **rectangle** for informational or parking signs 🅿️.

---

### 🚀 Application to Road Sign Recognition
By applying contour detection, we can:
1. 🛠️ Extract shapes from road sign images.
2. 🧩 Recognize signs based on their geometric boundaries.

This approach is especially critical for:
- 🚘 **Autonomous Driving Systems**: Providing real-time road sign detection.
- 🧭 **Driver Assistance Systems**: Enhancing navigation by recognizing road regulations.

---

### 💡 Goals
1. Implement and optimize contour detection techniques for road sign recognition.
2. Evaluate the accuracy and efficiency of this approach for various traffic signs.
3. Contribute to safer roads by advancing traffic sign recognition technology.

---

![Orange Brown Cute Illustrative Bingo Game Presentation (2)](https://github.com/user-attachments/assets/5930cb13-dc53-4907-b2e4-335203b09ec4)

### 📜 Abstract

The objective of this project is to develop a **computer vision system** for detecting and recognizing road safety signs or traffic signs using **contour detection techniques**. By focusing on the geometric shapes that define common road signs—such as:
- 🔴 **Circles** (e.g., Speed signs 🛑)
- 🔼 **Triangles** (e.g., Yield signs ⚠️)
- 🔷 **Rectangles** (e.g., Informational signs ℹ️)

We aim to create a robust method for extracting sign boundaries and identifying their types.

---

![Orange Brown Cute Illustrative Bingo Game Presentation (3)](https://github.com/user-attachments/assets/a0f08413-bcfe-4a3d-a59c-7230e7d51be5)

### 📂 Methodology
- 🛠️ **Preprocessing**: Enhancing road sign images for better contour detection.
- 🖼️ **Contour Detection**: Identifying boundaries of traffic signs.
- 🔍 **Shape Classification**: Determining the type of traffic sign based on its geometry.

### 📊 Dataset
The **Road Sign Detection dataset** will be utilized to validate the system's performance and ensure it can:
- ✅ Detect signs accurately.
- ✅ Classify signs into their respective types.

This framework integrates geometric recognition into a **traffic sign recognition system** designed to improve:
- 🚘 **Autonomous Driving Systems**
- 🧭 **Driver Assistance Technologies**

---

![Orange Brown Cute Illustrative Bingo Game Presentation (4)](https://github.com/user-attachments/assets/5aff4895-e619-470c-82e4-59cdfdd10eb7)

# 🚦 Conclusion 

This project successfully demonstrated the use of contour detection for traffic sign recognition, providing valuable insights and a foundation for further development in computer vision applications.  

---

## 📝 Findings  

- **🎯 Effectiveness of Contour Detection**:  
  The system accurately identified the geometric boundaries of common road signs, including circles, triangles, and rectangles, enabling reliable classification of:  
  - 🛑 Speed signs  
  - ⚠️ Yield signs  
  - 🅿️ Parking signs  

- **🚘 Practical Applications**:  
  The methodology shows strong potential for integration into:  
  - 🤖 Autonomous driving systems  
  - 🧭 Driver assistance technologies  
  Enhancing road safety and navigation.  

---

## ⚠️ Challenges  

- **🔍 Image Preprocessing**:  
  Variations in lighting, weather conditions, and sign visibility posed challenges in achieving consistent detection accuracy.  

- **❌ Shape Ambiguity or Form Uncertainty**:  
  Overlapping or damaged signs sometimes led to misclassification or caused errors in detection, highlighting the need for better image processing and advanced techniques to identify features accurately.
---

## 🏆 Outcomes  

- **✅ Improved Recognition Accuracy**:  
  The project delivered a precise and efficient system for detecting and classifying traffic signs, validated through the [Road Sign Detection dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/data).  

- **🚦 Advancement in Traffic Management**:  
  By demonstrating the practicality of contour detection, this work contributes to:  
  - Safer roadways  
  - Improved traffic systems  

This study underscores the importance of computer vision in modern transportation and serves as a stepping stone for innovations in real-time traffic sign recognition. 🌐 Further refinements, such as incorporating machine learning models alongside contour detection, could address challenges and enhance performance.  

---

![Orange Brown Cute Illustrative Bingo Game Presentation (5)](https://github.com/user-attachments/assets/7bf26f7b-7c1b-4b5f-b581-a0c254289bd4)

# 🚀 Code

## Outlining Shapes with Contours
### Apply the contour detection code to outline and highlight shapes in simple object images.
### 🛑 Outlining and Highlighting Road Signs Images Using Contour Detection 🚦
### Part 1: Import Libraries
 
 ```
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
 ``` 
### Part 2: Function to Detect and Outline Shapes

```
def detectAndOutlineShapes(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            # Detect triangles (3 vertices)
            if objCor == 3:
                cv2.drawContours(imgContour, [cnt], -1, (0, 255, 0), 4)
                cv2.putText(imgContour, "Triangle", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

            # Detect rectangles (4 vertices)
            elif objCor == 4:
                aspRatio = w / float(h)
                if 0.98 < aspRatio < 1.03:
                    cv2.drawContours(imgContour, [cnt], -1, (0, 0, 255), 4)
                    cv2.putText(imgContour, "Square", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.drawContours(imgContour, [cnt], -1, (255, 0, 0), 4)
                    cv2.putText(imgContour, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

            # Detect circles (more than 4 vertices)
            elif objCor > 4:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.drawContours(imgContour, [cnt], -1, (0, 255, 255), 4)
                cv2.putText(imgContour, "Circle", (int(x) - 20, int(y) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
```

### Example 1
### Part 3: Read Image and Preprocess 

```
path = 'Roadsign_Triangle.jpg'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

imgCanny = cv2.Canny(imgBlur, 50, 50)

```

### Part 4: Detect and Outline Shapes

```
detectAndOutlineShapes(imgCanny)

imgBlank = np.zeros_like(img)

imgStack = stackImages(0.8, ([img, imgGray, imgBlur],
                             [imgCanny, imgContour, imgBlank]))
```

### Part 5: Display the Outcome

```
cv2_imshow(imgStack)
cv2.waitKey(0)

```
![Screenshot 2024-12-11 170354](https://github.com/user-attachments/assets/d2c95c42-c003-48c2-a18e-8f24aa6a7566)

-1


### Example 2

```
path = 'Roadsign_Circle.jpg'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

imgCanny = cv2.Canny(imgBlur, 50, 50)

```
     
```
detectAndOutlineShapes(imgCanny)

imgBlank = np.zeros_like(img)

imgStack = stackImages(0.8, ([img, imgGray, imgBlur],
                             [imgCanny, imgContour, imgBlank]))

```
```
cv2_imshow(imgStack)
cv2.waitKey(0)
     
```
![Screenshot 2024-12-11 170420](https://github.com/user-attachments/assets/299498e4-7619-45ee-955d-0cbedcffe3ad)

-1

```
path = 'Roadsign_Rectangle.jpg'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

imgCanny = cv2.Canny(imgBlur, 50, 50)

```
     
```
detectAndOutlineShapes(imgCanny)

imgBlank = np.zeros_like(img)

imgStack = stackImages(0.8, ([img, imgGray, imgBlur],
                             [imgCanny, imgContour, imgBlank]))

```
     
```
cv2_imshow(imgStack)
cv2.waitKey(0)
     
```
![Screenshot 2024-12-11 170446](https://github.com/user-attachments/assets/2e66e000-24db-4296-8355-1934d3d7e101)

-1

---

![Orange Brown Cute Illustrative Bingo Game Presentation (7)](https://github.com/user-attachments/assets/c33aefff-0ad7-40fb-a6f8-21fcd78267c3)
**Join us in building smarter, safer roads through cutting-edge computer vision technology!** 🚦✨

---
# Reference 

## 🚦 Road Sign Detection Dataset  
This repository contains resources and analyses related to the **Road Sign Detection Dataset**, which is available on Kaggle. This dataset is ideal for training and testing machine learning models for road sign detection and recognition tasks.  

## 🔗 Access the Dataset:   
[Road Sign Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/data)  
Link: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/data

## 🔗 Video Guide for Code:  
Link: https://www.youtube.com/watch?v=Fchzk1lDt7Q

## 🔗 Open CV Guide for Code:
Link: https://www.youtube.com/watch?v=E3Lg4aZVCAU
