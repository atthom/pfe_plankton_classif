from pprint import pprint

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    # zca_whitening=True,
    zca_epsilon=1e-6,
    rotation_range=20,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.2,
    zoom_range=0.4,
    # channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=True,
    preprocessing_function=None,  # On pourrai peut être resize ici
    # data_format=K.image_data_format()
)


def create_database():
    database = dict()

    base_path = "./Dataset/train/"
    i = 2
    for dirname, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            np_matrix = img_to_array(load_img(dirname + "/" + filename))
            database[dirname + "/" + filename] = np_matrix
        i -= 1
        if i == 0:
            break
    return database


def gen_picture(path, img):
    dd = path.replace("Dataset/train", "Augmented_Train_Dataset")
    dd = dd.split("/")[0:len(dd.split("/")) - 1]
    dd = "/".join(dd)

    x = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dd,
                              save_prefix='plank_', save_format='jpg'):
        i += 1
        if i > 200:
            break  # otherwise the generator would loop indefinitely

# ./Augmented_Train_Dataset

print("creating database...")
database = create_database()
print("generating pictures...")
for path, img in database.items():
    gen_picture(path=path, img=img)
