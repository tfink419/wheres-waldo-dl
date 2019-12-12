from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from image_processing import process_img_to_bw, process_img

def LoadTrainData(dirname):
    images = []
    test_images = []
    labels = []
    test_labels = []
    for image in tqdm(os.listdir(dirname+'/notwaldo')[0:4000]):
        try:
            img = Image.open(dirname+'/notwaldo/'+image )
            img.load()
            asarr = np.asarray( img, dtype="int32" )
            images.append(list(asarr))
            img.close()
            labels.append([1.0, 0.0])
        except:
            print("failed image")
    for image in tqdm(os.listdir(dirname+'/notwaldo')[4001:-1]):
        try:
            img = Image.open(dirname+'/notwaldo/'+image )
            img.load()
            asarr = np.asarray( img, dtype="int32" )
            test_images.append(list(asarr))
            img.close()
            test_labels.append([1.0, 0.0])
        except:
            print("failed image")
    for image in tqdm(os.listdir(dirname+'/waldo')[0:5000]):
        try:
            img = Image.open(dirname+'/waldo/'+image )
            img.load()
            asarr = np.asarray( img, dtype="int32" )
            images.append(list(asarr))
            img.close()
            labels.append([0, 1.0])
        except:
            print("failed image")
    for image in tqdm(os.listdir(dirname+'/waldo')[5001:-1]):
        try:
            img = Image.open(dirname+'/waldo/'+image )
            img.load()
            asarr = np.asarray( img, dtype="int32" )
            test_images.append(list(asarr))
            img.close()
            test_labels.append([0, 1.0])
        except:
            print("failed image")
    images = np.asarray(images)
    test_images = np.asarray(test_images)
    labels = np.asarray(labels)
    test_labels = np.asarray(test_labels)
    # labels = np.expand_dims(labels, axis=2)
    return images, labels, test_images, test_labels


def LoadTestData(dirname):
    images = []
    image_names = []
    for image in tqdm(os.listdir(dirname)):
        try:
            img = Image.open(dirname+'/'+image )
            img.load()
            asarr = np.asarray( img, dtype="int32" )
            images.append(list(asarr))
            img.close()
            image_names.append(image)
        except:
            print("failed image")
    images = np.asarray(images)
    return images, image_names

def LoadOneBWTestData(image_location, chop_size):
    return process_img_to_bw(image_location, chop_size)

def LoadOneColorTestData(image_location, chop_size):
    return process_img(image_location, chop_size)
