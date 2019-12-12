from __future__ import print_function
import keras
from LoadData import LoadTrainData
from metrics import precision_m
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend, optimizers
import random

DATA_SET = '64-expanded'
BATCH_SIZE = 100
NUM_CLASSES = 1
EPOCHS = 100

# input image dimensions
img_rows, img_cols = 64, 64

train_images, train_labels, test_images, test_labels = LoadTrainData(DATA_SET)

print(train_images.shape, train_labels.shape)

# the data, split between train and test sets

if backend.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 3, img_rows, img_cols)
    test_images = test_images.reshape(test_images.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 3)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

train_images = train_images.astype('float32')
# test_images = test_images.astype('float32')
train_images /= 255
# test_images /= 255


# convert class vectors to binary class matrices
# train_labels_matrix = keras.utils.to_categorical(train_labels, NUM_CLASSES)
# test_labels_matrix = keras.utils.to_categorical(test_labels, NUM_CLASSES)

adam = optimizers.Adam(learning_rate=0.00002, beta_1=0.9, beta_2=0.999, amsgrad=False)

model = Sequential()
model.add(Conv2D(50, kernel_size=(2, 2),
                 activation='relu',
                 strides=(1,1),
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(25, kernel_size=(2, 2),
                 activation='relu',
                 strides=(1,1),
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(24, activation='sigmoid'))#, weights = [np.zeroes([400, 50]), np.zeroes(50)]))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', #binary_crossentropy, categorical_crossentropy, mean_squared_error
              optimizer=adam,
              metrics=['binary_accuracy', 'mae'])

history = model.fit(train_images, train_labels,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          # verbose=1, callbacks=[time_callback]),
          validation_data=(test_images, test_labels))
model.save('model-'+DATA_SET+'.h5')


plt.close()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


score = model.evaluate(test_images, test_labels, verbose=1)

# # predictions = model.predict_classes(test_images)
# predictions_matrix = model.predict(train_images)
#
print('Test accuracy:', score[1])
#
# conf_mat = tf.math.confusion_matrix(train_labels,predictions,num_classes=NUM_CLASSES)
# print("Confusion Matrix:", conf_mat)
#
# for i in range(NUM_CLASSES):
#     print("%s Results:" % (CLASS_NAMES[i]))
#     print("Accuracy: %f" % (accuracy_m(conf_mat, i)))
#     print("Precision: %f" % (precision_m(conf_mat, i)))
#     print("Recall: %f" % (recall_m(conf_mat, i)))
#     print("F1: %f\n" % (f1_m(conf_mat, i)))



# calculate accuracy and other metrics for each class
