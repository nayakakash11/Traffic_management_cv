import cv2
import cvzone
import math
from ultralytics import YOLO
from sort.sort import Sort
import numpy as np
import time

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Class names for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load masks for each video stream
mask1 = cv2.imread("mask2.png")
mask2 = cv2.imread("mask3.png")
mask3 = cv2.imread("mask4.png")
mask4 = cv2.imread("mask5.png")

# Tracking setup for four different roads
tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker3 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker4 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Capture video streams from four different roads
cap1 = cv2.VideoCapture("cars2.mp4")
cap2 = cv2.VideoCapture("cars3.mp4")
cap3 = cv2.VideoCapture("cars4.mp4")
cap4 = cv2.VideoCapture("cars5.mp4")

limits = [400, 297, 673, 297]
limits1 = [250, 370, 620, 370]
limits2 = [450, 700, 850, 700]
limits3 = [480, 510, 900, 510]
limits4 = [500, 500, 900, 500]
totalCount1, totalCount2, totalCount3, totalCount4 = [], [], [], []

# Initialize traffic light states for each road (0 = red, 1 = green)
traffic_light_states = [1, 0, 0, 0]  # Road 1 starts with a green light

# Time thresholds (in seconds)
min_green_time = 10
max_green_time = 30

# Time to keep the light green (in seconds) based on vehicle count
green_light_time = [0, 0, 0, 0]

# Last switch time to manage the timing
last_switch_time = time.time()


# Function to manage traffic light states and timing
def update_traffic_lights(counts):
    global last_switch_time, green_light_time, traffic_light_states, totalCount1, totalCount2, totalCount3, totalCount4

    # Get the current time
    current_time = time.time()

    # Find the road with the current green light
    for i in range(4):
        if traffic_light_states[i] == 1:  # Current road with green light
            elapsed_time = current_time - last_switch_time
            print(
                f"Road {i + 1} has green light. Elapsed time: {elapsed_time}, Green light time: {green_light_time[i]}")

            if elapsed_time >= green_light_time[i]:
                # Reset the vehicle count for the current road
                if i == 0:
                    totalCount1 = []
                elif i == 1:
                    totalCount2 = []
                elif i == 2:
                    totalCount3 = []
                elif i == 3:
                    totalCount4 = []

                # Move to the next road in round-robin fashion
                next_road = (i + 1) % 4

                # Set the traffic lights for the new road
                for j in range(4):
                    traffic_light_states[j] = 1 if j == next_road else 0

                # Recalculate green light time for the new road
                green_light_time[next_road] = max(min_green_time, min(counts[next_road] * 5, max_green_time))
                last_switch_time = current_time
                print(
                    f"Switching green light to road {next_road + 1}. New green light time: {green_light_time[next_road]}")

            break  # Exit after processing the current green light road


# Function to draw traffic light on each frame
def draw_traffic_light(img, state, position=(50, 50)):
    if state == 1:
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red
    cv2.circle(img, position, 20, color, cv2.FILLED)


def process_frame(cap, tracker, totalCount, limits, mask):
    success, img = cap.read()
    if not success:
        return None, totalCount

    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask_resized)

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(imgRegion, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(imgRegion, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)



    return img, totalCount


while True:
    # Process each video stream and update the vehicle counts
    img1, totalCount1 = process_frame(cap1, tracker1, totalCount1, limits1, mask1)
    img2, totalCount2 = process_frame(cap2, tracker2, totalCount2, limits2, mask2)
    img3, totalCount3 = process_frame(cap3, tracker3, totalCount3, limits3, mask3)
    img4, totalCount4 = process_frame(cap4, tracker4, totalCount4, limits4, mask4)

    # If any of the frames are not returned correctly, exit the loop
    if img1 is None or img2 is None or img3 is None or img4 is None:
        break

    # Update traffic lights based on the current vehicle counts and timing
    counts = [len(totalCount1), len(totalCount2), len(totalCount3), len(totalCount4)]
    update_traffic_lights(counts)

    # Draw the traffic lights on each corresponding image
    draw_traffic_light(img1, traffic_light_states[0], (100, 100))
    draw_traffic_light(img2, traffic_light_states[1], (100, 100))
    draw_traffic_light(img3, traffic_light_states[2], (100, 100))
    draw_traffic_light(img4, traffic_light_states[3], (100, 100))

    # Stack the images for display
    imgStack = cvzone.stackImages([img1, img2, img3, img4], 2, 0.5)

    # Show the stacked images
    cv2.imshow("Image", imgStack)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close windows
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()