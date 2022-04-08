import cv2
import numpy as np
from embedding import EmbeddingManager
from modelmanager import ModelManager

#data manager for handling recognizer vectors
embManager = EmbeddingManager("models/openface.nn4.small2.v1.t7")
embManager.setProfile("alex")
embFrameRate = 5;

#model manager for handling actual image processing
modelManager = ModelManager(
    "models/deploy.prototxt.txt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

print("Initialising Camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera!")
    exit()

frameId = 0 
while True:
    ret, frame = cap.read()
    frameId += 1

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    h, w = frame.shape[:2]
    
    modelManager.detectFace(frame)

    confidence = modelManager.getConfidence(0)
    if confidence > 0.7:
        (x, y, x1, y1) = modelManager.getFaceBounds(0, w, h)
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x1, y1), color=(0, 0, 255))

        roiFrame = frame[y:y1, x:x1]
        if (frameId % embFrameRate == 0):
            embManager.processFrame(roiFrame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()