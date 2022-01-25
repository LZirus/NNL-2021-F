import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import MaxPool2D, Conv2D, Input, Dense, Flatten, AveragePooling2D, Dropout, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast, RandomFlip, RandomRotation
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras import utils, models

# some useful variables

batch_size = 32                             # the amount of images processed at once
img_size = (180, 180)                       # img_size in pixels
img_size_vgg = (224, 224)                   # the vgg network needs a fixed img_size
def_epochs = 11                                 # default epochs used in training - should be chosen wisely
imgs_path = os.path.join('..', 'img')       # path to the dataset
num_classes = 3                             # default number of classes/ labels

# String for the correct usage of predict()
correct_usage = ('correct usage: \n' 
                'predict(\'category\', [model], [labels], img_path=[img_path]) \n'
                'predict(\'probabilities\', [model], [labels], img_path=[img_path]) \n'
                'predict(\'detection\', [model], [labels], img_path=[img_path])\n'
                'predict(\'live_detection\', [model], [labels])'
                'optional parameter: img_size, default=(180,180)')


####### FaceNet

# converts a loaded image into the correct format for the openCV face detection network
def convertImg(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# uses the FaceNet to localize faces in a picture
def faceNetLocalize(img, **kwargs):
    #parameters
    scaleFactor = kwargs.get('scaleFactor', 1.1)    # between 1.05 (quality) and 1.4 (speed) recommended (scale of the faces we search for)
    minNeighbors = kwargs.get('minNeighbors', 4)    # between 3 (quantity) and 6 (quality) recommended
    minSize = kwargs.get('minSize', (10, 10))       # min size of a face in the picture
    faceNet = kwargs.get('faceNet', init_faceNet()) # pretrained model from openCV
    
    # conversion and localization
    img_cvt = convertImg(img)
    return faceNet.detectMultiScale3(img_cvt, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, outputRejectLevels = True)

# initialize the network using openCV
def init_faceNet(**kwargs):
    path = kwargs.get('path', 'haarcascade_frontalface_default.xml')
    return cv2.CascadeClassifier(path)


######### MASK - CLASSIFIER

### get the untrained model structure by name

def select_model(model_name, **kwargs):
    #optional parameters
    num_labels = kwargs.get('num_classes', num_classes) # number of labels: is necessary for a correct output on the last Dense layer
    # num_classes automatically set in load_dataset

    # simplest model
    basic_model = Sequential()
    basic_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(180,180,3)))
    basic_model.add(AveragePooling2D(pool_size=(7,7)))
    basic_model.add(Flatten(name="flatten"))
    basic_model.add(Dense(128, activation="relu"))
    basic_model.add(Dropout(0.5)) #drops small confidences
    basic_model.add(Dense(num_labels, activation="softmax"))
    
    #more complicated model, utilizing 2 convolutional layers like the vgg - still very time-consuming in training
    small_model = Sequential()
    small_model.add(Conv2D(filters=128, kernel_size=(5,5), activation='relu', input_shape=(180,180,3)))
    small_model.add(Conv2D(filters=128, kernel_size=(5,5), activation='relu'))
    small_model.add(MaxPool2D(pool_size=(3,3)))
    small_model.add(Flatten(name="flatten"))
    small_model.add(Dense(units=224, activation="relu"))
    small_model.add(Dropout(0.5))#drops small confidences
    small_model.add(Dense(num_labels, activation="softmax"))
    
    #smaller version of the "vgg_small_model"
    vgg_smaller_model=Sequential()
    vgg_smaller_model.add(Conv2D(64,(3,3),activation='relu',input_shape=(180,180,3)))
    vgg_smaller_model.add(MaxPool2D(2,2))
    vgg_smaller_model.add(Conv2D(128,(3,3),activation='relu'))
    vgg_smaller_model.add(MaxPool2D(2,2))
    vgg_smaller_model.add(Flatten())
    vgg_smaller_model.add(Dropout(0.5))
    vgg_smaller_model.add(Dense(120,activation='relu'))
    vgg_smaller_model.add(Dense(num_labels,activation='softmax'))
    
    # simpler version of the vgg model, utilizes only one convolutional layer at a time, before max-pooling
    # but keeps the general design of the vgg model
    vgg_small_model=Sequential()
    vgg_small_model.add(Conv2D(64,(3,3),activation='relu',input_shape=(180,180,3)))
    vgg_small_model.add(MaxPool2D(2,2))
    vgg_small_model.add(Conv2D(64,(3,3),activation='relu'))
    vgg_small_model.add(MaxPool2D(2,2))
    vgg_small_model.add(Conv2D(128,(3,3),activation='relu'))
    vgg_small_model.add(MaxPool2D(2,2))
    vgg_small_model.add(Conv2D(128,(3,3),activation='relu'))
    vgg_small_model.add(MaxPool2D(2,2))
    vgg_small_model.add(Flatten())
    vgg_small_model.add(Dropout(0.5))
    vgg_small_model.add(Dense(120,activation='relu'))
    vgg_small_model.add(Dense(num_labels,activation='softmax'))


    # the vgg model is a state of the art model design, we do not have the necessary power or training data to utilize this model
    vgg_model = Sequential()
    vgg_model.add(Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3), padding="same", activation="relu", strides=(1,1))) 
    vgg_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    vgg_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    vgg_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(MaxPool2D(pool_size=(2, 2), strides=(2)))
    vgg_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(MaxPool2D(pool_size=(2, 2), strides=(2)))
    vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(MaxPool2D(pool_size=(2, 2), strides=(2)))
    vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg_model.add(MaxPool2D(pool_size=(2, 2), strides=(2)))
    vgg_model.add(Flatten())
    vgg_model.add(Dense(units=4096, activation="relu"))
    vgg_model.add(Dense(units=4096, activation="relu"))
    vgg_model.add(Dense(units=num_labels, activation="softmax"))
    

    # returns the correct model

    if model_name == 'basic_model':
        #basic_model.summary()
        return basic_model, img_size
    
    if model_name == 'small_model':
        #small_model.summary()
        return small_model, img_size

    if model_name == 'vgg_model':
        #vgg_model.summary()
        return vgg_model, img_size_vgg
    
    if model_name == 'vgg_small_model':
        #vgg_small_model.summary()
        return vgg_small_model, img_size
    
    if model_name == 'vgg_smaller_model':
        #vgg_smaller_model.summary()
        return vgg_smaller_model, img_size
    
    return vgg_small_model, img_size



##### load images 

def load_dataset(**kwargs):
    global num_classes
    
    print("Loading Dataset")
    # optional values
    imgs_path = kwargs.get('imgs_path', os.path.join('..', 'img'))
    img_size = kwargs.get('img_size', (180, 180))

    # initializing lists
    valid_images = [".jpg",".png",".jpeg",".JPG"]
    x=[]
    y=[]
    
    # loop over files in image directory
    for root, dirs, files in os.walk(imgs_path):
        for filename in files:
            end = os.path.splitext(filename)[1]
            if end.lower() not in valid_images:
                continue
            image = load_img(os.path.join(root, filename), target_size=img_size)
            image = img_to_array(image)
            
            label = os.path.join(root, filename).split(os.path.sep)[-2]
            
            x.append(image)
            y.append(label)
    
    # convert images and labels to arrays for further use
    x = np.array(x, dtype="float32")
    y = np.array(y)

    ## convert labels to ints from 0 ... len(labels)-1
    labels = []
    for i in range(len(y)):
        try:
            j = labels.index(y[i])
        except:
            labels.append(y[i])
            j = labels.index(y[i])
        y[i] = j
    y.astype(int)
    
    num_classes = len(labels)
    print(num_classes, "classes")
    
    ## split dataset
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=3)

    ## one-hot encoding
    trainY = utils.to_categorical(trainY, num_classes)
    testY = utils.to_categorical(testY, num_classes)

    ## data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    ## merge xs and ys
    train_batches = datagen.flow(trainX, trainY, batch_size=32, subset='training')
    test_batches  = datagen.flow(trainX, trainY, batch_size=32, subset='validation')
    
    return train_batches, test_batches, labels, testX, testY, trainX, trainY

### use loaded images to train specified model

def train_model(model, train_ds, val_ds, **kwargs):
    # optional parameter
    epochs = kwargs.get('epochs', def_epochs)   # training epochs
    
    # compile the model
    model.compile(
        loss='mean_squared_error', 
        optimizer='adam', 
        metrics=['accuracy'])
    
    # train/fit the model and save the history for documentation/ testing purposes
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, shuffle=True)
    return history

######### Load/ Save model

def save_model(path, model):
    model.save(path, save_format="h5")
    
def load_model_good(path):
    return models.load_model(path)


########### Prediction/ Application part

# returns the label with the highest confidence of a prediction
def maskPredict(model, img, labels):
    pred = model.predict(img[None])
    label_index = np.argmax(pred)
    print(labels[label_index])
    return labels[label_index], pred[0][label_index]


#mode can be 'category', 'probabilities', 'detection', 'live_detection'
def predict(mode, model, **kwargs):
    #model = kwargs.get('model', load_model())
    img_path = kwargs.get('img_path', None)
    img_size = kwargs.get('img_size', (180, 180))
    labels = kwargs.get('labels', None)
    
    # display the help menu 
    if mode=='help':
        print(correct_usage)
        return

    # change to live_detection and interpret camera feed
    if mode=='live_detection':
        live_det(model, img_size, labels)
        return
    
    # print help menu, as function was malused
    if img_path is None:
        print(correct_usage)
        return

    #load_img
    img = load_img(img_path, target_size = img_size)
    img = img_to_array(img)
    
    #change into detection mode
    if mode=='detection':
        detect(img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
        
    # change into category mode, printing the label of highest confidence
    if mode=='category':
        img = img / 255
        label, _ = maskPredict(model, img, labels)
        return label
        
    # change into probability mode, printing the probability for each label
    if mode=='probabilities':
        img = img / 255
        confidences = model.predict(img[None])[0]
        ret = {}
        for lab in range(len(labels)):
            ret[labels[lab]] = confidences[lab]
        # ret_str = ''
        # for lab in range(len(labels)):
        #   ret_str = ret_str + (f"{labels[lab]}: {confidences[lab]}")
        # return ret_str
        return ret

    # no valid mode was specified: print help menu
    else:
        print(correct_usage)

# detect and classify a face on an image
def detect(model, img, img_size, labels):
    faceLocs, rejectLevels, confidences = faceNetLocalize(img)
        
    for (x, y, w, h) in faceLocs:
        #crop image and predict label of cropped image
        img_crop = img[y:y+w, x:x+h]
        img_crop = cv2.resize(img_crop, img_size)
        img_crop = img_crop / 255
        label, confidence_mask = maskPredict(model, img_to_array(img_crop), labels)

        #show label/ bounding box on image
        cv2.putText(img, f"{label}, confidence:{confidence_mask}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2) 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    cv2.imshow('live_output', img)

# detect and classify images from a live-feed
def live_det(model, img_size, labels):
    wait_time = 10 #time in ms to wait before refreshing feed
    camera = cv2.VideoCapture(0) #Input value might differ on different systems
    
    while(True):
        ret, img = camera.read()
        if not ret:
            print('Error: failed reading camera')
            return 'Error: failed reading camera'
        detect(model, img, img_size, labels)

        #wait for ESC or q
        if (cv2.waitKey(wait_time) & 0xFF) in [27, ord('q')]: 
            break

    camera.release()
    cv2.destroyAllWindows()
    return 'live_output'


