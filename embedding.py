from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import cv2
from sklearn import svm


class EmbedderContext:
    _state      = None
    _svm        = None
    _embedder   = None
    _labels     = []
    _embeddings = []

    _resultName = ""
    _resultProb = 0.0

    profileDir  = Path("./profiles/")
    _profiles   = []
    _currentProfile = None

    def __init__(self, embedderPath) -> None:
        print("Loading embedder model... ")
        self.embedder = cv2.dnn.readNetFromTorch(embedderPath)

        print("Loading vector machine...")
        self.svm = svm.SVC(kernel="rbf", probability=True)

        self.setState(LoadProfilesState())
        self.process()

    def setState(self, state: EmbedderState):
        self._state = state
        self._state.ctx = self

    def process(self, frame=None):
        self._state.process(frame)

    @property
    def resultName(self):
        return self._resultName

    @property
    def resultProb(self):
        return self._resultProb

    @property
    def uLabels(self):
        return np.unique(self._labels)

    @property
    def svm(self):
        return self._svm

    @svm.setter
    def svm(self, instance):
        if instance:
            self._svm = instance
        else:
            print("SVM cannot be null")

    @property
    def embedderPath(self):
        return self._embedderPath

    @property
    def embedder(self):
        return self._embedder

    @embedder.setter
    def embedder(self, instance):
        if instance:
            self._embedder = instance
        else:
            print("Embedder cannot be null")

    @property
    def profiles(self):
        return self._profiles

    @property
    def currentProfile(self):
        return self._currentProfile

    @currentProfile.setter
    def currentProfile(self, name):
        if not self.__getProfileByName(name):
            newprofile = Profile(name, self.profileDir.joinpath(name))
            self.addProfile(newprofile)
            self._currentProfile = newprofile
            print("Profile {} added and selected".format(name))
        else:
            self._currentProfile = self.__getProfileByName(name)
            print("Profile {} selected".format(name))

        self.setState(CollectingSamplesState())

    def addProfile(self, profile):
        if not self.__getProfileByName(profile.name):
            self.profiles.append(profile)
        else:
            print("Profile already exists")

    def __getProfileByName(self, name):
        return next((d for d in self.profiles if d.name == name), None)


class EmbedderState(ABC):

    @property
    def ctx(self) -> EmbedderContext:
        return self._ctx

    @ctx.setter
    def ctx(self, context: EmbedderContext) -> None:
        self._ctx = context

    @abstractmethod
    def process(self, frame) -> None:
        pass


class LoadProfilesState(EmbedderState):
    def process(self, frame) -> None:
        print("Loading profiles...")

        try:
            for path in self.ctx.profileDir.iterdir():
                if path.is_dir():
                    images = [img for img in self.ctx.profileDir.joinpath(
                        path.name).iterdir()]
                    profile = Profile(path.name, path, images)
                    self.ctx.addProfile(profile)
        except:
            print("Could not load profiles")

        profileCount = len(self.ctx.profiles)
        print("Loaded {} profiles".format(profileCount))

        self.ctx.setState(CollectingSamplesState())


class CollectingSamplesState(EmbedderState):
    def process(self, frame) -> None:
        if not self.ctx.currentProfile.hasEnoughData():
            try:
                count = self.ctx.currentProfile.addFrame(frame)
                print("Collected {}/{} samples".format(count,
                            self.ctx.currentProfile.sampleSize))
            except:
                print("Error while writing frame")
        else:
            print("Enough samples collected. Normalising...")
            self.ctx.setState(NormaliseState())


class NormaliseState(EmbedderState):
    def process(self, frame) -> None:
        print("Normalising samples...")

        dataset = []
        for idx, profile in enumerate(self.ctx.profiles):
            for fid, file in enumerate(profile.data):
                roiFrame = cv2.imread(str(file))
                frameBlob = cv2.dnn.blobFromImage(roiFrame, 1 / 255.0, (96, 96), (0, 0, 0))
                self.ctx.embedder.setInput(frameBlob)

                frameVec = self.ctx.embedder.forward().flatten()
                dataset.append((idx, frameVec))
                print("Normalised sample {}[{}/{}]".format(profile.name, fid+1, profile.sampleSize))

        (Y, X) = zip(*dataset)
        self.ctx._labels = Y
        self.ctx._embeddings = X
        self.ctx.setState(TrainSvmState())

class TrainSvmState(EmbedderState):
    def process(self, frame) -> None:
        if (len(self.ctx.uLabels) >= 2):
            self.ctx.svm.fit(self.ctx._embeddings, self.ctx._labels)
            print("SVM data fitted")
            self.ctx.setState(AnalyseState())
        else:
            print("Not enough profiles to train SVM")

class AnalyseState(EmbedderState):
    def process(self, frame) -> None:
        frameBlob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (96, 96), (0, 0, 0))
        self.ctx.embedder.setInput(frameBlob)
        frameVec = self.ctx.embedder.forward()

        p = self.ctx.svm.predict_proba(frameVec)[0]
        j = np.argmax(p)

        self.ctx._resultName = self.ctx.profiles[j].name
        self.ctx._resultProb = p[j]

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
        val = len(self.data)

        if self.hasEnoughData():
            return val

        name = "sample{}.png".format(val)

        path = self.dir.joinpath(name + "/")

        cv2.imwrite(str(path), frame)

        self.data.append(path)

        return val
