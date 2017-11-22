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

print("creating database...")
database = create_database()
print("generating pictures...")
for path, img in database.items():
    gen_picture(path=path, img=img)
