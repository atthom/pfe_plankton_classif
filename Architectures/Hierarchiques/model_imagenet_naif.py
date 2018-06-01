import numpy as np
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_data_dir = "./uvp5ccelter_group1/"
nb_train_samples = 200
nb_validation_samples = 20
epochs = 3
batch_size = 16
# dimensions of our images.
resolution = (150, 150)
nb_img = 662668


def create_model():
    img_input = Input(shape=(150, 150, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Decision block
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(40, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='vgg16')

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
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

    train_generator = datagen.flow_from_directory(
        train_data_dir, batch_size=batch_size,
        target_size=resolution, color_mode='grayscale')

    model.fit_generator(train_generator,
                        steps_per_epoch=100,
                        #steps_per_epoch= nb_img //b batch_size,
                        validation_data=train_generator,
                        validation_steps= 20,
                        #validation_steps= nb_img // batch_size,
                        epochs=epochs, workers=4)

    save_model(model, "VGG16_model_naif_without_detritus.h5")


print("Compiling model...")
model = create_model()
print("Training...")
fit_data(model)
