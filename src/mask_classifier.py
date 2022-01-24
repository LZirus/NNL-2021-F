import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPool2D, Conv2D, Input, Dense, Flatten, AveragePooling2D, Dropout
import tensorflow.keras.layers as lays
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomContrast, RandomFlip, RandomRotation
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Sequential, losses as lfs
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import keras_tuner as kt
from tensorflow.keras.callbacks import ModelCheckpoint

augmentation = Sequential([
  RandomFlip("horizontal"),
  RandomRotation(0.4),
  RandomContrast(0.5)
])
batch_size = 32
img_size = (180, 180)
img_size_vgg = (224, 224)
epochs = 11
checkpoint_path = "mask_model/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
imgs_path = os.path.join('..', 'img')
num_classes = 2

correct_usage=  'correct usage: \n' + 'predict([path to image], \'category\' \n' + 'predict([path to image], \'probabilities\' \n' + 'predict([path to image], \'detection\' \n' + 'predict(\'live_detection\')'

def select_model(model_name, **kwargs):
    num_labels = kwargs.get('num_classes', num_classes)

    basic_model = Sequential([ 
    Rescaling(1. /255),
    augmentation,
    Conv2D(32, (3,3), activation='relu'),
    AveragePooling2D(pool_size=(7,7)),
    Flatten(name="flatten"),
    Dense(128, activation="relu"),
    Dropout(0.5),#drops small confidences
    Dense(num_labels, activation="softmax")
    ])

    small_model = Sequential([ 
        Rescaling(1. /255),
        augmentation,
        Conv2D(filters=128, kernel_size=(5,5), activation='relu'),
        Conv2D(filters=128, kernel_size=(5,5), activation='relu'),
        MaxPool2D(pool_size=(3,3)),
        Flatten(name="flatten"),
        Dense(units=224, activation="relu"),
        Dropout(0.5),#drops small confidences
        Dense(num_labels, activation="softmax")
        ])

    vgg_small_model = Sequential([ 
        Rescaling(1. /255),
        Conv2D(64, (3,3), activation='relu'),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3,3), activation='relu'),
        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(name="flatten"),
        Dense(256, activation="relu"),
        Dropout(0.5),#drops small confidences
        Dense(num_labels, activation="softmax")
        ])

    vgg_model = Sequential([
        Rescaling(1. /255),
        Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3), padding="same", activation="relu", strides=(1,1)), 
        Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2)),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2)),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2)),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2)),
        Flatten(),
        Dense(units=4096, activation="relu"),
        Dense(units=4096, activation="relu"),
        Dense(units=num_labels, activation="softmax")
    ])

    if model_name == 'basic_model':
        return basic_model
    
    if model_name == 'small_model':
        return small_model

    if model_name == 'vgg_model':
        return vgg_model
    
    if model_name == 'vgg_small_model':
        return vgg_small_model

##FaceNet

def convertImg(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def faceNetLocalize(img, **kwargs):
    scaleFactor = kwargs.get('scaleFactor', 1.1)    #between 1.05 (quality) and 1.4 (speed) recommended (scale of the faces we search for)
    minNeighbors = kwargs.get('minNeighbors', 4)    #between 3 (quantity) and 6 (quality) recommended
    minSize = kwargs.get('minSize', (10, 10))       #min size of a face in the picture
    faceNet = kwargs.get('faceNet', init_faceNet())
    
    img_cvt = convertImg(img)
    return faceNet.detectMultiScale3(img_cvt, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, outputRejectLevels = True)

def init_faceNet(**kwargs):
    path = kwargs.get('path', 'haarcascade_frontalface_default.xml')
    return cv2.CascadeClassifier(path)

##Mask - Classifier

def load_dataset(**kwargs):
    imgs_path = kwargs.get('imgs_path', os.path.join('..', 'img'))
    img_size = kwargs.get('img_size', (180, 180))
    batch_size = kwargs.get('batch_size', 32)

    train_ds = image_dataset_from_directory(imgs_path,  validation_split=0.2, subset="training",  seed=3, image_size=img_size,  batch_size=batch_size)
    val_ds = image_dataset_from_directory(imgs_path,  validation_split=0.2, subset="validation",  seed=3, image_size=img_size,  batch_size=batch_size)
    labels = train_ds.class_names

    y_test = np.concatenate([y for _, y in val_ds], axis=0)
    x_test = np.concatenate([x for x, _ in val_ds], axis=0)
    return train_ds, val_ds, labels, y_test, x_test

def train_model(model, train_ds, val_ds, **kwargs):
    epochs = kwargs.get('epochs', 10)
    checkpoint_path = kwargs.get('checkpoint_path', "mask_model/weights.ckpt")

    #Callback to save model's weights
    #https://www.tensorflow.org/tutorials/keras/save_and_load
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose = 1)

    model.compile(optimizer=Adam(0.01), loss=lfs.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback])
    return history

def evaluate_model(x_test, y_test, model):
    results = model.evaluate(x_test, y_test, batch_size=32)
    print(results)
    
    y_pred_confidences = model.predict(x_test)
    y_pred = [np.argmax(cs) for cs in y_pred_confidences]
    print(classification_report(y_test, y_pred))

def save_model(path, model):
    model.save(path, save_format="h5")

def load_model(path):
    return models.load_model(path)

def load_model(**kwargs):
    checkpoint_path = kwargs.get('checkpoint_path', "mask_model/weights.ckpt")
    model = kwargs.get('model', select_model('basic_model'))
    model.load_weights(checkpoint_path)
    return model

def maskPredict(model, img, labels):
    pred = model.predict(img[None])
    label_index = np.argmax(pred)
    return labels[label_index], pred[0][label_index]


#mode can be 'category', 'probabilities', 'detection', 'live_detection'
def predict(mode, **kwargs):
    img_path = kwargs.get('img_path', None)
    model = kwargs.get('model', load_model())
    if mode=='live_detection':
        live_det()
        return
    
    if img_path is None:
        print(correct_usage)
        return

    #load_img
    img = load_img(img_path, target_size = img_size)
    img = img_to_array(img)
    
    if mode=='detection':
        detect(img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
        
    if mode=='category':
        label, confidence = maskPredict(img)
        return label
        
    if mode=='probabilities':
        #TODO: schöneres Format
        return model.predict(img[None])

    else:
        print(correct_usage)

def detect(img):
    faceLocs, rejectLevels, confidences = faceNetLocalize(img)
        
    for (x, y, w, h) in faceLocs:
        #crop image and predict label of cropped image
        img_crop = img[y:y+h, x:x+w]
        label, confidence_mask = maskPredict(img)
        #show label/ bounding box on image
        cv2.putText(img, f"{label}, confidence:{confidence_mask}", (x+w-30, y+h), cv2.FONT_HERSHEY_PLAIN, 1.0, cv2.CV_RGB(0,255,0), 2.0) 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('live_output', img)

def live_det():
    #TODO: errorhandling for camera
    
    wait_time = 10 #time in ms to wait before refreshing feed
    camera = cv2.VideoCapture(0) #Input value might differ on different systems
    
    while(True):
        _, img = camera.read()

        detect(img)

        #wait for ESC or q
        if (cv2.waitKey(wait_time) & 0xFF) in [27, ord('q')]: 
            break

    camera.release()
    return 'live_output'
