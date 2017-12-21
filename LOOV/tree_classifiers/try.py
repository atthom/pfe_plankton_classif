from keras.preprocessing.image import ImageDataGenerator
from TreeClassifier import TreeClassifier
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, save_model, load_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input, losses, metrics
from keras.optimizers import Adam
import numpy as np
import os
import json
from PIL import Image

separator = os.sep

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

super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif"
# super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\Dataset\\PFE"
# classifier = TreeClassifier(super_path)

# classifier.create_achitecture(datagen, nb_epoch=10)

# path_classif = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif\\living\\meduses\\Hydrozoa"
# path_classif = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif\\fiber"
path_classif = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif\\living\\Copecope\\Copepoda"

# path_classif = "E:\Polytech_Projects\pfe_plankton_classif\LOOV\super_classif\living\globe_avec_points\Rhizaria"

# classifier.classify(path_classif)


def create_top_model(nb_classes):
    input_tensor = Input(shape=(150, 150, 3))
    base_model = applications.VGG16(
        include_top=False, weights='imagenet', input_tensor=input_tensor)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    adam = Adam(lr=0.000001)
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=adam, metrics=[metrics.categorical_accuracy])

    return model


def train_model(datagen, data_dir, nb_epoch):
    # model = self.create_model(len(os.listdir(data_dir)))
    model = create_top_model(len(os.listdir(data_dir)))
    batch_size = 16
    nb_img = sum([len(files) for r, d, files in os.walk(data_dir)])

    train_generator = datagen.flow_from_directory(
        data_dir, batch_size=8, target_size=(150, 150), color_mode='rgb')

    model.fit_generator(train_generator,
                        # steps_per_epoch=nb_img // batch_size,
                        steps_per_epoch=100,
                        validation_data=train_generator,
                        # validation_steps=nb_img // (2 * batch_size),
                        validation_steps=20,
                        epochs=nb_epoch, workers=4)
    data_dir = data_dir.split(separator)[-1]
    save_model(model, "trry/model_" + data_dir + ".h5")

    # Writing JSON data
    with open("./trry/labels_" + data_dir + ".json", "w") as f:
        json.dump(train_generator.class_indices, f)


def load_architecture(directories):
    for data_dir in directories:
        dd = data_dir.split(separator)[-1]
        model = load_model("./trry/model_" + dd + ".h5")
        with open("./trry/labels_" + dd + ".json", "r") as f:
            label = dict((v, k) for k, v in json.load(f).items())
    return model, label


def classify(super_path, path):
    print("load achitecture...")
    model, label = load_architecture([super_path])

    for ll in os.listdir(path)[0:5]:
        print("img", ll, "...")
        # this is a PIL image
        img = load_img(path + separator + ll, grayscale=True)
        img = img.resize((150, 150), Image.ANTIALIAS)
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)
        print("classify...")
        print(model.predict(img, batch_size=8, verbose=0))
        print(label[model.predict_classes(
            img, batch_size=1, verbose=0)[0]])


train_model(datagen, super_path, 3)
print(os.listdir(super_path))

classify(super_path, path_classif)
