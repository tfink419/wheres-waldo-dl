from keras import backend
import tensorflow as tf
import time
import keras

def precision_m(y_true, y_pred):
  true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
  predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + backend.epsilon())
  return precision
