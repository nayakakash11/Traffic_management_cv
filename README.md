# Traffic_management_cv

## Overview

This project is a Smart Traffic Management System that utilizes computer vision and YOLOv8 object detection to monitor and control traffic flow at an intersection. The system dynamically adjusts traffic signals based on real-time vehicle counts to optimize traffic efficiency.

## Features

Vehicle Detection & Tracking: Uses YOLOv8 and SORT algorithm for tracking cars, trucks, buses, and motorbikes.

Adaptive Traffic Signal Control: Adjusts signal duration based on traffic density.

Multi-Camera Support: Processes video feeds from four different roads.

Masking for Region of Interest: Applies pre-defined masks to filter detection areas.

Real-time Visualization: Displays live traffic monitoring with bounding boxes and vehicle counts.

## Setup Instructions

Clone the repository:

git clone: https://github.com/nayakakash11/Traffic_management_cv.git

Install dependencies:

pip install -r requirements.txt

Download the YOLOv8 model weights and place them in the Yolo-Weights folder.

Add dataset/video files in the project directory: https://drive.google.com/drive/folders/1zQOGq0fz2fdyHWEgLpYpKrX_EsNQdzK1?usp=sharing
