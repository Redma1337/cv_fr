import cv2
import numpy

class ModelManager:
    blobConfig = { 
        "dim": (300, 300), #dimensions of the frame our model is trained on
        "scale": 1.0, #scaling factor for the frame
        "meanVals": (104.0, 117.0, 123.0) #mean values used for serialization
    }

    def __init__(self, caffeProto, caffeModel):
        print("Loading caffe model...")
        self.net = cv2.dnn.readNetFromCaffe(caffeProto, caffeModel)

        if not self.net:
            print("Failed to load caffe model")

    def detectFace(self, frame, debug=False):
        """
        Use opencv serialization and the caffe model to detect faces in a frame. Results are stored in data, facesCount Attributes.
        
        Arguments:
            frame {Mat} -- opencv class handeling frame data
            debug {bool} -- determines if blob output should be written to frame (default: {False})
        """
        resizedFrame = cv2.resize(frame, self.blobConfig["dim"])
        frameBlob = cv2.dnn.blobFromImage(resizedFrame, self.blobConfig["scale"], self.blobConfig["dim"], self.blobConfig["meanVals"])
        
        if debug:
            cv2.putText(frame, "F_B: {}".format(frameBlob.shape), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

        self.net.setInput(frameBlob)

        self.data = self.net.forward()
        self.facesCount = self.data.shape[2]

    def getConfidence(self, index):
        if not self.data.any():
            print("No loaded data found...")
        
        return self.data[0, 0, index, 2]

    def getFaceBounds(self, index, width, height):
        """
        Returns Coordinates of the face in the given dimensions.
        
        Arguments:
            index {int} -- index of the face to get bounds for
            width {int} -- width of the frame
            height {int} -- height of the frame
        """
        if not self.data.any():
            print("No loaded data found...")
        
        box = self.data[0, 0, index, 3:7] * numpy.array([width, height, width, height])

        return box.astype("int")

    