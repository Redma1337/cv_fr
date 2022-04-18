from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import cv2
from sklearn import svm


class EmbedderContext:
    _state      = None #current state of state machine
    _svm        = None
    _embedder   = None 
    _labels     = [] #labels used in svm
    _embeddings = [] #features used in our svm
    _reqUpdate  = True #determines if we need to normalize our data aigan

    _resultProb = 0.0

    profileDir  = Path("./profiles/")
    _profiles   = []
    _currentProfile = None

    def __init__(self, embedderPath) -> None:
        if not self.profileDir.exists():
            self.profileDir.mkdir()

        print("Loading embedder model... ")
        self.embedder = cv2.dnn.readNetFromTorch(embedderPath)

        print("Loading vector machine...")
        self.svm = svm.SVC(kernel="rbf", probability=True)

        #first state is loading profiles
        self.setState(LoadProfilesState())
        self.process()

    def setState(self, state: EmbedderState):
        self._state = state
        self._state.ctx = self

    def process(self, frame=None):
        self._state.process(frame)

    @property
    def resultProb(self):
        """Returns the probability of the current frame being the current face, estimated by the Analyse State"""
        return self._resultProb

    @property
    def uLabels(self):
        """Returns labels without dublicates"""
        return np.unique(self._labels)

    @property
    def svm(self):
        """Returns the svm instance we are training"""
        return self._svm

    @svm.setter
    def svm(self, instance):
        if instance:
            self._svm = instance
        else:
            print("SVM cannot be null")

    @property
    def embedderPath(self):
        """Returns the path to the embedder model"""
        return self._embedderPath

    @property
    def embedder(self):
        """Returns the embedder instance used to extract features"""
        return self._embedder

    @embedder.setter
    def embedder(self, instance):
        if instance:
            self._embedder = instance
        else:
            print("Embedder cannot be null")

    @property
    def profiles(self):
        """Returns array of profiles currently loaded"""
        return self._profiles

    @property
    def currentProfile(self):
        """Returns the current profile recognized"""
        return self._currentProfile

    @currentProfile.setter
    def currentProfile(self, name):
        """Sets the current profile to the one with the given name"""
        if not self._currentProfile == None and name == self._currentProfile.name:
            return

        if not self.__getProfileByName(name):
            newprofile = Profile(name, self.profileDir.joinpath(name))
            self.addProfile(newprofile)
            self._currentProfile = newprofile

            #since we added new profiles, also set the flag to retrain our model
            self._reqUpdate = True
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
    """
    This state loads profiles according to the folders found in the profiles directory.
    Also instantly recognizes how many samples these Profiles alsready have collected.

    Next State: CollectingSamplesState
    """

    def process(self, frame) -> None:
        print("Loading profiles...")

        try:
            for path in self.ctx.profileDir.iterdir():
                if path.is_dir():
                    images = [img for img in self.ctx.profileDir.joinpath(path.name).iterdir()]
                    profile = Profile(path.name, path, images)
                    self.ctx.addProfile(profile)
        except:
            print("Could not load profiles")

        profileCount = len(self.ctx.profiles)
        print("Loaded {} profiles".format(profileCount))

        self.ctx.setState(CollectingSamplesState())


class CollectingSamplesState(EmbedderState):
    """
    Collects missing samples if profile has sufficient data to use for training.

    Next State: NormaliseState
    """
    def process(self, frame) -> None:
        if not self.ctx.currentProfile.hasEnoughData():
            try:
                count = self.ctx.currentProfile.addFrame(frame)
                print("Collected {}/{} samples".format(count,
                            self.ctx.currentProfile.sampleSize))
            except:
                print("Error while writing frame")
        else:
            print("Enough samples collected.")

            #skip normalise states if we dont have new samples
            if (self.ctx._reqUpdate):
                self.ctx.setState(NormaliseState())
            else:
                self.ctx.setState(TrainSvmState())

class NormaliseState(EmbedderState):
    """
    This state will turn collected samples into 128-d vector using embeddor model which is then used for training.

    Next state: TrainSvmState
    """

    def process(self, frame) -> None:
        print("Normalising samples...")

        dataset = []
        for idx, profile in enumerate(self.ctx.profiles):
            for fid, file in enumerate(profile.data):
                #extract frame from sample.png
                roiFrame = cv2.imread(str(file))
                frameBlob = cv2.dnn.blobFromImage(roiFrame, 1 / 255.0, (96, 96), (0, 0, 0))
        
                #give sample into model to convert
                self.ctx.embedder.setInput(frameBlob)

                #retrieve the 128-d vec result of the sample
                frameVec = self.ctx.embedder.forward().flatten()
                dataset.append((idx, frameVec))
                print("Normalised sample {}[{}/{}]".format(profile.name, fid+1, profile.sampleSize))

        (Y, X) = zip(*dataset)
        self.ctx._labels = Y
        self.ctx._embeddings = X
        self.ctx._reqUpdate = False
        self.ctx.setState(TrainSvmState())

class TrainSvmState(EmbedderState):
    """
    This state will train the SVM with the collected and normalised sampledata/profile labels.

    Next state: AnalyseState
    """
    def process(self, frame) -> None:
        if (len(self.ctx.uLabels) >= 2):
            #using support vector machines (svm) provided by sklearn to train svm instance with our data
            self.ctx.svm.fit(self.ctx._embeddings, self.ctx._labels)

            print("SVM data fitted")
            self.ctx.setState(AnalyseState())
        else:
            print("Not enough profiles to train SVM")

class AnalyseState(EmbedderState):
    """
    This state will analyse the current frame and try to identify the current face.
    """
    def process(self, frame) -> None:
        #get the 128-d fec of current frame
        frameBlob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (96, 96), (0, 0, 0))
        self.ctx.embedder.setInput(frameBlob)
        frameVec = self.ctx.embedder.forward()

        #run it through our model
        predicitons = self.ctx.svm.predict_proba(frameVec)[0]
        guessedIndex = np.argmax(predicitons)

        self.ctx.currentProfile = self.ctx.profiles[guessedIndex].name
        self.ctx._resultProb = predicitons[guessedIndex]

class Profile:
    def __init__(self, name, dir, data=[], sampleSize=10):
        self.name = name
        self.data = data
        self.sampleSize = sampleSize

        self.dir = dir
        if not dir.exists():
            dir.mkdir()

    def __repr__(self):
        return "({}, {}, {})".format(self.name, self.dir, len(self.data))

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
