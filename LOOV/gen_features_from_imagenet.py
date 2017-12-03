import numpy as np
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

nb_train_samples = 200
nb_validation_samples = 20
epochs = 30
batch_size = 16
# dimensions of our images.
resolution = (150, 150)
nb_img = 662668

train_data_dir = "/home/user/Project/pfe_plankton_classif/Dataset/level0"
train_data_dir = "/home/tjalaber/pfe_plankton_classif/Dataset/uvp5ccelter/level0"

def create_model():
    input_tensor = Input(shape=(150, 150, 3))
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    return base_model


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
        target_size=resolution, color_mode='rgb',
        class_mode=None, shuffle=False)

    bottleneck_features_train = model.predict_generator(generator, nb_img // (batch_size), verbose=1)
    np.save(open('bottleneck_features.npy', 'wb'), bottleneck_features_train)

    save_model(model, "VGG16_imagenet_features.h5")


print("Compiling model...")
model = create_model()
print("Training...")
fit_data(model)
