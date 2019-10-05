import cv2
import numpy as np
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
import time

from PIL import Image
from imutils import paths
import os

seed = 42
np.random.seed(seed)

# datapath = './cmfd_forge/casia/'
datapath1 = './forge/coco/authentic/'
datapath2 = './forge/coco/tamper/'
datapath3 = './splicing/'

imagePaths1 = list(paths.list_images(datapath1))
imagePaths2 = list(paths.list_images(datapath2))

# print(len(imagePaths1))
# print(len(imagePaths2))

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


data = np.array(data)
labels = np.array(labels)

# print(labels)

print(data.shape)
print(labels.shape)

np.save('x_new.npy', data)
np.save('y_new.npy', labels)

print("Done.....")
