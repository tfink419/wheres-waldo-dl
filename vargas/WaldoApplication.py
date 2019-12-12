import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow import keras
from skimage import io
import time
from keras.optimizers import SGD

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

train_images = []
train_labels = []
val_images = []
val_labels = []
test_images = []
test_labels = []

not_waldo_files =  os.listdir('64/notwaldo')
waldo_files = os.listdir('64/waldo')

rand_train_notwaldo = np.random.choice(not_waldo_files, 4000)
rand_train_waldo = np.random.choice(waldo_files, 4000)
rand_val_notwaldo = np.random.choice(not_waldo_files, 500)
rand_val_waldo = np.random.choice(waldo_files, 500)
rand_test_notwaldo = np.random.choice(not_waldo_files, 500)
rand_test_waldo = np.random.choice(waldo_files, 500)

for file_name in rand_train_notwaldo:
    try:
        img = io.imread('64/notwaldo/'+ file_name)
        train_images.append(img)
        train_labels.append(0)
    except:
        print("failed image")

for file_name in rand_train_waldo:
    try:
        img = io.imread('64/waldo/'+ file_name)
        train_images.append(img)
        train_labels.append(1)
    except:
        print("failed image")

for file_name in rand_val_notwaldo:
    try:
        img = io.imread('64/notwaldo/'+ file_name)
        val_images.append(img)
        val_labels.append(0)
    except:
        print("failed image")

for file_name in rand_val_waldo:
    try:
        img = io.imread('64/waldo/'+ file_name)
        val_images.append(img)
        val_labels.append(1)
    except:
        print("failed image")

for file_name in rand_test_notwaldo:
    try:
        img = io.imread('64/notwaldo/'+ file_name)
        test_images.append(img)
        test_labels.append(0)
    except:
        print("failed image")

for file_name in rand_test_waldo:
    try:
        img = io.imread('64/waldo/'+ file_name)
        test_images.append(img)
        test_labels.append(1)
    except:
        print("failed image")

train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
train_data_norm = (train_images/255)
val_data_norm = (val_images/255)
test_data_norm = (test_images/255)
random.seed(554433)
fixed = list(zip(train_data_norm,train_labels))
fixedVal = list(zip(val_data_norm,val_labels))
fixedTest = list(zip(test_data_norm,test_labels))
random.shuffle(fixed)
random.shuffle(fixedVal)
train_data_norm, train_labels = zip(*fixed)
train_data_norm = np.array(train_data_norm)
val_data_norm, val_labels = zip(*fixedVal)
val_data_norm = np.array(val_data_norm)
test_data_norm, test_labels = zip(*fixedTest)
test_data_norm = np.array(test_data_norm)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5,5), strides=1, activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
time_callback = TimeHistory()
history = model.fit(train_data_norm, train_labels, batch_size=100, epochs=200, verbose=1, shuffle='batch', callbacks=[time_callback], validation_data=(val_data_norm, val_labels))

results = model.predict_classes(test_data_norm, batch_size=100, verbose=1)
results = results.flatten()
temp = sum(test_labels == results)
print("Test Accuracy: ", temp/len(test_labels))
incorrectIndices = np.array(np.nonzero(results != test_labels))
incorrectIndices = incorrectIndices.flatten()
incorrectIndex = incorrectIndices[0]
incorrect_image_arr = test_data_norm[incorrectIndex]
plt.imshow(incorrect_image_arr)
plt.show()
incorrectIndex = incorrectIndices[1]
correct_image_arr = test_data_norm[incorrectIndex]
plt.imshow(correct_image_arr)
plt.show()


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()