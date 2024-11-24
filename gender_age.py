#To Run follow the guide given: 
# Step 1 : Execute the following command to detect from image: python gender_age.py --input path/to/image.jpg --output path/to/save/detected/images 
# Step 2 : Execute the following command to detect from video: python gender_age.py --input path/to/video.mp4 --output path/to/save/detected/videos
# Step 3 : Execute the following command to detect from live webcam: python gender_age.py --output ./results



# Import required modules
import cv2 as cv
import math
import time
import argparse
import os

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# Argument Parser
parser = argparse.ArgumentParser(description="Run age and gender detection using OpenCV's DNN module.")
parser.add_argument("-i", "--input", type=str, help="Path to input image or video file. Skip to use the webcam.")
parser.add_argument("-o", "--output", type=str, default="./detected", help="Directory to save detected images. Default is './detected'.")

args = parser.parse_args()

# Pretrained models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Age and Gender Categories
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
ageNet = cv.dnn.readNetFromCaffe(ageProto, ageModel)
genderNet = cv.dnn.readNetFromCaffe(genderProto, genderModel)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Check input and output paths
if args.input and not os.path.isfile(args.input):
    print(f"Error: Input file '{args.input}' not found.")
    exit(1)

if not os.path.exists(args.output):
    os.makedirs(args.output)

# Open input (file or webcam)
cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20

while cv.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("End of input.")
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face detected. Checking next frame...")
        continue

    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1), max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender Prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f"Gender: {gender}, Confidence: {genderPreds[0].max():.3f}")

        # Age Prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f"Age: {age}, Confidence: {agePreds[0].max():.3f}")

        label = f"{gender}, {age}"
        cv.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    # Display the result
    cv.imshow("Age and Gender Detection", frameFace)

    # Save the result
    if args.input:
        output_path = os.path.join(args.output, os.path.basename(args.input))
        cv.imwrite(output_path, frameFace)
        print(f"Output saved to {output_path}")

    print(f"Processing Time: {time.time() - t:.3f}s")
