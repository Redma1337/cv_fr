from sklearn import svm
from sklearn.svm import LinearSVC
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm

class EmbeddingManager:
    BLOB_CONFIG = { 
        "dim": (96, 96),
        "scale": 1.0 / 255,
        "meanVals": (0, 0, 0)
    }

    PROFILE_DIR = Path("./profiles/")

    def __init__(self, embeddorPath):
        self.profiles = list()
        self.__loadProfiles()
        self.normalised = False

        print("Loading embedder model... ")
        self.embedder = cv2.dnn.readNetFromTorch(embeddorPath)

        if not self.embedder:
            print("Failed to load embedder model")
            exit()

    def __loadProfiles(self):
        for f in self.PROFILE_DIR.iterdir(): 
            if f.is_dir():
                profile = Profile(f.name, f, [t for t in self.PROFILE_DIR.joinpath(f.name).iterdir()])
                self.profiles.append(profile)

    def setProfile(self, name):
        if not self.getProfileByName(name):
            newprofile = Profile(name, self.PROFILE_DIR.joinpath(name))
            self.profiles.append(newprofile)
            self.currentprofile = newprofile
        else:
            self.currentprofile = self.getProfileByName(name)

        self.normalised = False
        print(self.profiles)

    def getProfileByName(self, name):
        return next((d for d in self.profiles if d.name == name), None)

    def processFrame(self, frame):
        if not self.currentprofile.hasEnoughData():
            self.currentprofile.addFrame(frame)
        elif not self.normalised:
            self.__trainSVM(self.__normaliseShards())
        else:
            frameBlob = cv2.dnn.blobFromImage(frame, self.BLOB_CONFIG["scale"], self.BLOB_CONFIG["dim"], self.BLOB_CONFIG["meanVals"])
            self.embedder.setInput(frameBlob)

            frameVec = self.embedder.forward()
            p = self.vectorMachine.predict_proba(frameVec)[0]
            j = np.argmax(p)
            print(j, self.labels)

    def __normaliseShards(self):
        dataset = []
        for idx, profile in enumerate(self.profiles):
            for file in profile.data:
                roiFrame = cv2.imread(str(file))
                frameBlob = cv2.dnn.blobFromImage(roiFrame, self.BLOB_CONFIG["scale"], self.BLOB_CONFIG["dim"], self.BLOB_CONFIG["meanVals"])
                self.embedder.setInput(frameBlob)

                frameVec = self.embedder.forward()
                dataset.append((idx, frameVec.flatten()))

        self.normalised = True
        print("Normalised data...")
        return dataset

    def __trainSVM(self, dataset):
        (Y, X) = zip(*dataset)
        self.labels = Y;
        clf = svm.SVC(kernel="rbf", probability=True)
        clf.fit(X, Y)
        
        self.vectorMachine = clf
        print("Plotted SVC...")


class Profile:
    def __init__(self, name, dir, data=[], sampleSize=10):
        self.name = name
        self.data = data
        self.sampleSize = sampleSize

        self.dir = dir
        if not dir.exists():
            dir.mkdir()

    def __repr__(self):
        return "Name: {} | Dir: {} | Data ({})\n".format(self.name, self.dir, len(self.data))

    def hasEnoughData(self):
        return len(self.data) >= self.sampleSize

    def addFrame(self, frame):
        if self.hasEnoughData(): return

        val = len(self.data)
        name = "shard{}.png".format(val)

        path = self.dir.joinpath(name + "/")

        cv2.imwrite(str(path), frame)

        self.data.append(path)