from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_data_dir = "./uvp5ccelter_group1/"
batch_size = 16
# dimensions of our images.
resolution = (150, 150)

train_data_dir = "E:\Polytech_Projects\pfe_plankton_classif\LOOV\super_classif"


def take_model():
    return load_model("E:/Polytech_Projects/pfe_plankton_classif/LOOV/naive_minimal.h5")
 # VGG16_model_naif_without_detritus.h5")


def validate(model):
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
        train_data_dir, target_size=resolution,
        batch_size=batch_size, color_mode='grayscale',
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

    predicted = model.predict_generator(generator, 2000)

    y_true = []
    for i in range(40):
        y_true += [i] * (2000 // 40)

    y_true = np.array(y_true)
    y_pred = predicted > 0.5
    cm = confusion_matrix(y_true, y_pred)

    #y_true = np.array([0] * 1000 + [1] * 1000)
    #y_pred = probabilities > 0.5

    #confusion_matrix(y_true, y_pred)


print("load model...")
model = take_model()
print("validate model...")
validate(model)
