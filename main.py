import cv2
import numpy as np
from embedding import EmbedderContext
from modelmanager import ModelManager
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

#data manager for handling recognizer vectors
embCtx = EmbedderContext(config["DEFAULT"]["EMBEDDING_MODEL_PATH"])

#setting the current profile which would be selected by the car
embCtx.currentProfile = "alex"

#rate in which samples are being taken
embFrameRate = 5;

#model manager for handling actual image processing
modelManager = ModelManager(
    config["DEFAULT"]["CAFFE_PROTO_PATH"],
    config["DEFAULT"]["CAFFE_MODEL_PATH"],
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
    
    #only get one face since we only assume one driver
    modelManager.detectFace(frame)

    #assure our guess was somewhat exact
    confidence = modelManager.getConfidence(0)
    if confidence > 0.7:
        (x, y, x1, y1) = modelManager.getFaceBounds(0, w, h)
        cv2.rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(x1, y1), 
            color=(0, 0, 255)
        )

        #to make samples more accurate we are cutting the face region out
        roiFrame = frame[y:y1, x:x1]

        if (frameId % embFrameRate == 0):
            embCtx.process(roiFrame)

        cv2.putText(frame, "{} {:.2f}%".format(embCtx.resultName, embCtx.resultProb), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()