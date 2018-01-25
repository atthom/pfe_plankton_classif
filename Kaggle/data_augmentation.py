from pprint import pprint

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(119))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


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
    rescale=False,
    preprocessing_function=None,  # On pourrai peut être resize ici
    # data_format=K.image_data_format()
)


def train_model(model):
    nb_train_samples = 2000
    nb_validation_samples = 800

    train_data = np.load('bottleneck_features_train.npy')
    #A = np.zeros((int(nb_train_samples / 2),))
    #B = np.ones((int(nb_train_samples / 2),))
    #train_labels = np.concatenate((A, B))
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    # train_labels = np.array(        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load('bottleneck_features_validation.npy')
    #A = np.zeros((int(nb_validation_samples / 2),))
    #B = np.ones((int(nb_validation_samples / 2),))
    #validation_labels = np.concatenate((A, B))
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    # validation_labels = np.array(        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model.fit(train_data, train_labels, epochs=50, batch_size=16,
              validation_data=(validation_data, validation_labels))
    model.save_weights('bottleneck_fc_model.h5')


def gaussianize(img):
    from scipy.ndimage.filters import gaussian_filter

    blurred = gaussian_filter(img, sigma=7)


def generate_pictures():
    dir_names = [_ for _ in os.listdir(base_path)]
    print("Generating pictures...")
    for dirname in dir_names:
        print(dirname)
        files = [_ for _ in os.listdir(base_path + dirname)]

        lenfiles = len(files)

        i = 0
        for file in files:
            i += 1
            print("File n°", i, "sur", lenfiles, "...")
            path = base_path + dirname + "/" + file
            img = img_to_array(load_img(path))
            gen_picture(path, img)


def create_database():
    database = dict()

    base_path = "./Dataset/train/"

    dir_names = [_ for _ in os.listdir(base_path)]

    for dir_name in dir_names:
        files = [_ for _ in os.listdir(base_path + dir_name)]

        for file in files:
            path = base_path + dir_name + "/" + file
            img = img_to_array(load_img(path))  # this is a PIL image
            # this is a Numpy array with shape (3, 150, 150)
            database[path] = img.reshape((1,) + img.shape)

    return database


def gen_picture(path, img):
    dd = path.replace("Dataset/train", "Augmented_Train_Dataset")
    print(path, dd)
    dd = dd.split("/")[0:len(dd.split("/")) - 1]
    dd = "/".join(dd)
    if not os.path.isdir(dd):
        os.mkdir(dd)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=dd,
                              save_prefix='plank_', save_format='jpg'):
        i += 1
        if i > 200:
            break  # otherwise the generator would loop indefinitely


# ./Augmented_Train_Dataset

# print("creating database...")
# database = create_database()
# print("generating pictures...")
# for path, img in database.items():
#    gen_picture(path=path, img=img)

train_path = "./Dataset/train/"
test_path = "./Dataset/train/"
new_train_path = "./Dataset/Augmented_Train_Dataset/"
new_test_path = "./Dataset/Augmented_Test_Dataset/"
# dir_names = [_ for _ in os.listdir(base_path)]


train_generator = datagen.flow_from_directory(train_path, target_size=(150, 150),
                                              color_mode='grayscale', class_mode='binary',
                                              batch_size=16, save_to_dir=new_train_path,
                                              save_prefix='plank_', save_format='jpg')

validation_generator = datagen.flow_from_directory(new_train_path, target_size=(150, 150),
                                                   color_mode='grayscale', class_mode='categorical',
                                                   batch_size=16, save_to_dir=new_test_path,
                                                   save_prefix='plank_', save_format='jpg')

model = create_model()
train_model(model)

# bottleneck_features_train = model.predict_generator(train_generator, steps=2000,
#                                                    max_queue_size=64, workers=8,
#                                                    use_multiprocessing=True)
# np.save('bottleneck_features_train.npy', bottleneck_features_train)


# bottleneck_features_validation = model.predict_generator(train_generator, steps=800,
#                                                         max_queue_size=64, workers=8)
# np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

# model.save_weights('model.h5')
