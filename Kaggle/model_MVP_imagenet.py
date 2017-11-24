import numpy as np
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_data_dir = "./Dataset/train/"
validation_data_dir = "./Dataset/train/"
nb_train_samples = 200
nb_validation_samples = 20
epochs = 30
batch_size = 8
# dimensions of our images.
resolution = (150, 150)
nb_img = 29715


def create_model():
    input_tensor = Input(shape=(150, 150, 3))
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    model = Sequential()

    model.add(base_model)

    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(119, activation='softmax'))

    # top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    # model = Model(input=base_model.input, output=top_model(base_model.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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
        target_size=resolution, color_mode='rgb')

    validation_generator = datagen.flow_from_directory(
        validation_data_dir, target_size=resolution,
        batch_size=batch_size, color_mode='rgb')  # , color_mode='grayscale')

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_img // (4 * batch_size),
                        validation_data=validation_generator,
                        validation_steps=nb_img // (8 * batch_size),
                        epochs=30, workers=5)

    save_model(model, "VGG16_model.h5")


print("Compiling model...")
model = create_model()
print("Training...")
fit_data(model)
