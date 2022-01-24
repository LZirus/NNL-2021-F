basic_model = Sequential([ 
    Rescaling(1. /255),
    augmentation,
    Conv2D(32, (3,3), activation='relu'),
    AveragePooling2D(pool_size=(7,7)),
    Flatten(name="flatten"),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
    ])