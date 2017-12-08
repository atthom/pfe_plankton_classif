from keras.preprocessing.image import ImageDataGenerator
from LayerClassifier import LayerClassifier

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
classifier = LayerClassifier(super_path)

#classifier.create_achitecture(datagen, nb_epoch=10)
