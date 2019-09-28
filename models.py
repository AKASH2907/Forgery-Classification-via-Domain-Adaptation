import numpy as np
from keras.layers import Input, Dense, Activation, BatchNormalization, PReLU, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelBinarizer
import matplotlib
matplotlib.use("Agg")
import datetime

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.applications import ResNet50
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.initializers import RandomNormal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import pickle
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils, plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121



def vgg_16(inputs):
	print("VGG 16 model loaded ......")
	x = Conv2D(64, (3, 3), padding='same')(inputs)
	x = Activation('relu')(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(64, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layer 3 & 4
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(128, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(128, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layer 5, 6, & 7
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), padding='same')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layers 8, 9, & 10
	# x = ZeroPadding2D((1, 1))(x)
	# x = Conv2D(512, (3, 3), padding='same')(x)
	# x = Activation('relu')(x)
	# x = ZeroPadding2D((1, 1))(x)
	# x = Conv2D(512, (3, 3), padding='same')(x)
	# x = Activation('relu')(x)
	# x = ZeroPadding2D((1, 1))(x)
	# x = Conv2D(512, (3, 3), padding='same')(x)
	# x = Activation('relu')(x)
	# x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layers 11, 12, & 13
	# x = ZeroPadding2D((1, 1))(x)
	# x = Conv2D(512, (3, 3), padding='same')(x)
	# x = Activation('relu')(x)
	# x = ZeroPadding2D((1, 1))(x)
	# x = Conv2D(512, (3, 3), padding='same')(x)
	# x = Activation('relu')(x)
	# x = ZeroPadding2D((1, 1))(x)
	# x = Conv2D(512, (3, 3), padding='same')(x)
	# x = Activation('relu')(x)
	# x = MaxPooling2D(pool_size=(2, 2))(x)

	return x

# def vgg_19(inputs):

def alexnet(inputs):
	print("AlexNet model loaded ......")
	x4 = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid')(inputs)
	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(x4)

	x4 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='valid')(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(x4)

	x4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)

	x4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)

	x4 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x4)

	return x4

def lenet(inputs):

	x = Conv2D(20, (5, 5), padding="same")(inputs)
	x = Activation("relu")
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

	x = Conv2D(50, (5, 5), padding="same")
	x = Activation("relu")
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

	return x
