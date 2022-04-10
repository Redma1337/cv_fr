import cv2
import numpy as np
from embedding import EmbeddingManager
from modelmanager import ModelManager

#data manager for handling recognizer vectors
embManager = EmbeddingManager("models/openface.nn4.small2.v1.t7")

#setting the current profile which woul