import numpy as np
from keras.layers import Input, Dense, Activation, BatchNormalization, PReLU, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, f1_score

from sklearn.preprocessing import LabelBinarizer
import matplotlib
matplotlib.use("Agg")
import datetime
import tensorflow as tf

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
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
from models import alexnet, vgg_16

init = RandomNormal(mean = 0., stddev = 0.02)

Xs = np.load('x_new.npy')
ys = np.load('y_new.npy')

Xt = np.load('x_target.npy')
yt = np.load('y_target.npy')

y_s = []
y_t = []
for i in range(ys.shape[0]):
# for i in range(10):

	if ys[i]=='authentic':
		y_s+=[0]
	else:
		y_s+=[1]
for i in range(yt.shape[0]):
	if yt[i]=='authentic':
		y_t+=[0]
	else:
		y_t+=[1]

y_s = np.asarray(y_s)
y_t = np.asarray(y_t)

# print(np.unique(y_s))
print(y_s.shape, y_t.shape)
print(Xs.shape, ys.shape)
print(Xt.shape, yt.shape)


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def build_models(model_name):
    """Creates three different models, one used for source only training, two used for domain adaptation"""
    inputs = Input(shape=(224, 224, 3)) 

    if model_name=='alexnet':
    	x4 = alexnet(inputs)

    elif model_name=='vgg16':
        x4 = vgg_16(inputs)


    x4 = Flatten()(x4)

    x4 = Dense(32, activation='relu')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("elu")(x4)  

    source_classifier = Dense(2, activation='softmax', name="mo")(x4)

    # Domain Classification
    domain_classifier = Dense(16, activation='relu', name="do4")(x4)
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Activation("elu", name="do6")(domain_classifier)
    domain_classifier = Dropout(0.5)(domain_classifier)

    domain_classifier = Dense(2, activation='softmax', name="do")(domain_classifier)

    # Combined model
    comb_model = Model(inputs=inputs, outputs=[source_classifier, domain_classifier])
    comb_model.compile(optimizer="Adam",
              loss={'mo': 'categorical_crossentropy', 'do': 'categorical_crossentropy'},
              loss_weights={'mo': 1, 'do': 2}, metrics=['accuracy'], )

    source_classification_model = Model(inputs=inputs, outputs=[source_classifier])
    source_classification_model.compile(optimizer="Adam",
              loss={'mo': 'categorical_crossentropy'}, metrics=['accuracy'], )


    domain_classification_model = Model(inputs=inputs, outputs=[domain_classifier])
    domain_classification_model.compile(optimizer="Adam",
                  loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])
    
    
    embeddings_model = Model(inputs=inputs, outputs=[x4])
    embeddings_model.compile(optimizer="Adam",loss = 'categorical_crossentropy', metrics=['accuracy'])
                        
                        
    return comb_model, source_classification_model, domain_classification_model, embeddings_model


def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


def train(Xs, ys, Xt, yt,  enable_dann = True, n_iterations = 50):
    
    batch_size = 64
    
    # writer = tf.summary.FileWriter('./graph', )

    model, source_classification_model, domain_classification_model, embeddings_model = build_models('alexnet')

    log_path = './graph'
    callback = TensorBoard(log_path)
    callback.set_model(model)
    train_names = ['train_loss', 'train_mae']


#     y_class_dummy = np.ones((len(Xt), 2))
    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))    
    
    sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    sample_weights_adversarial = np.ones((batch_size * 2,))
    
    S_batches = batch_generator([Xs, to_categorical(ys)], batch_size)
    T_batches = batch_generator([Xt, np.zeros(shape = (len(Xt),2))], batch_size)
    
    for i in range(n_iterations):
        y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))
                
        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)

        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, np.zeros_like(y0)])

        adv_weights = []
        for layer in model.layers:
            if (layer.name.startswith("do")):
                adv_weights.append(layer.get_weights())

        if(enable_dann):
            # note - even though we save and append weights, the batchnorms moving means and variances
            # are not saved throught this mechanism 
            stats = model.train_on_batch(X_adv, [y_class, y_adversarial_1],
                                     sample_weight=[sample_weights_class, sample_weights_adversarial])
            
            write_log(callback, train_names, stats, i)
            
            # summary = tf.Summary(value=[tf.Summary.Value(tag="loss", 
            #                                  simple_value=value), ])
            # writer.add_summary(summary)

            k = 0
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    layer.set_weights(adv_weights[k])
                    k += 1

            class_weights = []
            
        
            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    class_weights.append(layer.get_weights())
            
            stats2 = domain_classification_model.train_on_batch(X_adv, [y_adversarial_2])

            k = 0
            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    layer.set_weights(class_weights[k])
                    k += 1

        else:
            stats = source_classification_model.train_on_batch(X0,y0)
            
       
        if ((i + 1) % 5 == 0):
            print(i, stats)
            y_test_hat_t = source_classification_model.predict(Xt).argmax(1)
            y_test_hat_s = source_classification_model.predict(Xs).argmax(1)
            print("Iteration %d, source accuracy =  %.5f, target accuracy = %.5f"%(i, accuracy_score(ys, y_test_hat_s), accuracy_score(yt, y_test_hat_t)))
            print("Iteration %d, source f1_score =  %.5f, target f1_score = %.5f"%(i, f1_score(ys, y_test_hat_s), f1_score(yt, y_test_hat_t)))
    return embeddings_model


embs = train(Xs, y_s, Xt, y_t, enable_dann=True)
