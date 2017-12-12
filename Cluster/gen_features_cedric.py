import numpy as np
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input, losses, metrics
from keras.optimizers import Adam
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_data_dir = "./uvp5ccelter_group1/"
nb_train_samples = 200
nb_validation_samples = 20
epochs = 30
batch_size = 16
# dimensions of our images.
resolution = (150, 150)
nb_img = 556376

nb_classes = 49

train_data_dir = "/home/user/Project/pfe_plankton_classif/Dataset/level0"
train_data_dir = "/home/tjalaber/pfe_plankton_classif/Dataset/uvp5ccelter/level0"
#train_data_dir = "E:\Polytech_Projects\pfe_plankton_classif\LOOV\super_classif"
train_data_dir = "E:\\Polytech_Projects\\pfe_plankton_classif\\Dataset\\train"


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(150, 150, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    adam = Adam(lr=0.000001)
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=adam, metrics=[metrics.categorical_accuracy])
    return model


def fit_data(model):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        # zca_whitening=True,
        # zca_epsilon=1e-6,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        # channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=None  # On pourrai peut être resize ici
        # data_format=K.image_data_format()
    )

    generator = datagen.flow_from_directory(
        train_data_dir, batch_size=batch_size,
        target_size=resolution, color_mode='grayscale',
        class_mode=None, shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, nb_img // (batch_size), verbose=1)
    np.save(open('bottleneck_features_cedric.npy', 'wb'),
            bottleneck_features_train)

    save_model(model, "model_feature_cedric.h5")


print("Compiling model...")
model = create_model()
print("Training...")
fit_data(model)
