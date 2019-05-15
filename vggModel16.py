import numpy as np 
import pandas as pd 

import os
# print(os.listdir("../Project/chest_xray/"))

import keras
import h5py
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Softmax,Input,Flatten
from keras.optimizers import Adam,RMSprop,SGD
from keras.layers.merge import add
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import BatchNormalization

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score
from keras.metrics import categorical_accuracy
# %matplotlib inline
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

#from sklearn.metrics import confusion_matrix
#from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger



model = Sequential()

model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', input_shape=(224,224,1), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.60))
model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

model.summary()


gen = ImageDataGenerator()
train_batches = gen.flow_from_directory("../Project/dataset/train",model.input_shape[1:3],color_mode="grayscale",shuffle=True,seed=1, batch_size=16)
valid_batches = gen.flow_from_directory("../Project/dataset/val", model.input_shape[1:3],color_mode="grayscale", shuffle=True,seed=1,batch_size=16)
#test_batches = gen.flow_from_directory("../Project/dataset/test", model.input_shape[1:3], shuffle=False,
 #                                      color_mode="grayscale", batch_size=8)

#train_batches = gen.flow_from_directory("../Project/chest_xray/train",model.input_shape[1:3],color_mode="grayscale",shuffle=True,seed=1,
 #                                               batch_size=16)
#valid_batches = gen.flow_from_directory("../Project/chest_xray/val", model.input_shape[1:3],color_mode="grayscale", shuffle=True,seed=1,
#                                                batch_size=16)
#test_batches = gen.flow_from_directory("../Project/chest_xray/test", model.input_shape[1:3], shuffle=False,
#                                               color_mode="grayscale", batch_size=8)

model.compile(Adam(lr=0.005, decay=1e-5),loss="categorical_crossentropy", metrics=["accuracy"])

csv_logger = CSVLogger('lognoaug.csv', append=True, separator=';')

history_pretrained=model.fit_generator(train_batches,validation_data=valid_batches,epochs=30, steps_per_epoch=15, validation_steps=16,callbacks=[csv_logger])

no_steps = len(test_batches)
p = model.predict_generator(test_batches, steps=no_steps, verbose=True)
pre = pd.DataFrame(p)
pre["filename"] = valid_batches.filenames
pre["label"] = (pre["filename"].str.contains("unhealthy")).apply(int)
pre['pre'] = (pre[1]>0.5).apply(int)
accuracy_score(pre["label"], pre["pre"])


# from sklearn.metrics import confusion_matrix
#CM = confusion_matrix(pre["label"], pre["pre"])
# from mlxtend.plotting import plot_confusion_matrix
#fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))
#plt.show()


#tn, fp, fn, tp = CM.ravel()

#precision = tp/(tp+fp)
#recall = tp/(tp+fn)

#print("Recall of the model is {:.2f}".format(recall))
#print("Precision of the model is {:.2f}".format(precision))


# summarize history for accuracy
plt.plot(history_pretrained.history["loss"],'r-x', label="Train Loss")
plt.plot(history_pretrained.history["val_loss"],'b-x', label="Validation Loss")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Pretrained'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_pretrained.history["acc"],'r-x', label="Train Accuracy")
plt.plot(history_pretrained.history["val_acc"],'b-x', label="Validation Accuracy")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Pretrained'], loc='upper left')
plt.show()




#model.compile(optimizer='rmsprop',
#                      loss='categorical_crossentropy',
#                                    metrics=['accuracy'])


#history_pretrained=model.fit_generator(train_batches,validation_data=valid_batches,epochs=30, steps_per_epoch=16, validation_steps=16)

#no_steps = len(test_batches)
#p = model.predict_generator(test_batches, steps=no_steps, verbose=True)
#pre = pd.DataFrame(p)
#pre["filename"] = test_batches.filenames
#pre["label"] = (pre["filename"].str.contains("PNEUMONIA")).apply(int)
#pre['pre'] = (pre[1]>0.5).apply(int)
#accuracy_score(pre["label"], pre["pre"])
