import cv2
import numpy as np
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from shutil import copyfile, move
import time

from PIL import Image
from imutils import paths
import os
from sklearn.preprocessing import LabelBinarizer
import matplotlib
matplotlib.use("Agg")
import math
import datetime

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import pickle
from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121

seed = 42
np.random.seed(seed)

# datapath = './cmfd_forge/casia/'
datapath1 = './forge/coco/authentic/'
datapath2 = './forge/coco/tamper/'
datapath3 = './splicing/'

imagePaths1 = list(paths.list_images(datapath1))
imagePaths2 = list(paths.list_images(datapath2))
imagePaths3 = list(paths.list_images(datapath3))

print(len(imagePaths1))
print(len(imagePaths2))

data = []
labels = []

for imagePath in imagePaths1[:5000]:

	label = imagePath.split(os.path.sep)[-2]
	# print(label)
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	data.append(image)
	labels.append(label)

print("COCO authentic done ....")

for imagePath in imagePaths2[:5000]:

	label = imagePath.split(os.path.sep)[-2]
	# print(label)
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	data.append(image)
	labels.append(label)

print("COCO tampering done ....")


for imagePath in imagePaths3:
	label = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	data.append(image)
	labels.append(label)
	
print("Splicing done ....")

data = np.array(data)
labels = np.array(labels)

# print(labels)

print(data.shape)
print(labels.shape)

np.save('x_new.npy', data)
np.save('y_new.npy', labels)

print("Done.....")
# perform one-hot encoding on the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

# print(labels)