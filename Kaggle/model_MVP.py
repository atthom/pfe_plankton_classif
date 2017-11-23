from pprint import pprint

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications

top_model_weights_path = 'vgg16_weights.h5'
train_data_dir = "./Dataset/train/"
validation_data_dir = "./Dataset/train/"
nb_train_samples = 500
nb_validation_samples = 200
epochs = 10
batch_size = 16
# dimensions of our images.
resolution = (150, 150)


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
        train_data_dir,  color_mode='grayscale',
        target_size=resolution, batch_size=batch_size)

    validation_generator = datagen.flow_from_directory(
        validation_data_dir, target_size=resolution,
        batch_size=batch_size, color_mode='grayscale')

    model.fit_generator(train_generator,
                        steps_per_epoch=2000 // batch_size,
                        epochs=50,
                        validation_data=validation_generator,
                        validation_steps=800 // batch_size)
    model.save_weights('naive_try.h5')


def create_model():
    # build the VGG16 network
    # model = applications.VGG16(include_top=False, weights='imagenet')

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(119, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def train_top_model(model):
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * int(nb_train_samples / 2) +
        [1] * int(nb_train_samples / 2))

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) +
        [1] * int(nb_validation_samples / 2))

    model.fit(train_data, train_labels,
              epochs=epochs, batch_size=batch_size, verbose=1,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)


print("Compile model...")
model = create_model()
print("Generate data and fit the model...")
fit_data(model)
print("Training...")
train_top_model(model)
