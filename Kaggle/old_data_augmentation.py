from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

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


def generate_pictures():
    train_data_dir = "./Dataset/train/"

    dir_names = [_ for _ in os.listdir(train_data_dir)]
    print("Generating pictures...")
    for dirname in dir_names:
        print(dirname)
        files = [_ for _ in os.listdir(train_data_dir + dirname)]
        lenfiles = len(files)

        i = 0
        for file in files:
            i += 1
            print("File n°", i, "sur", lenfiles, "...")
            path = train_data_dir + dirname + "/" + file
            img = load_img(path)  # this is a PIL image
            img = img.resize(resolution, Image.ANTIALIAS)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            gen_picture(path, img)


def gen_picture(path, img):
    dd = path.replace("train", "Augmented_Train_Dataset")
    print(path, dd)
    dd = dd.split("/")[0:len(dd.split("/")) - 1]
    dd = "/".join(dd)
    if not os.path.isdir(dd):
        os.makedirs(dd)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=dd,
                              save_prefix='plank_', save_format='jpg'):
        i += 1
        if i > 200:
            break  # otherwise the generator would loop indefinitely

batch_size = 16
# dimensions of our images.
resolution = (150, 150)

generate_pictures()
