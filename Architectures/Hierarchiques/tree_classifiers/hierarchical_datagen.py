import os
import numpy as np
import shutil
import glob
import PIL
from keras.preprocessing.image import ImageDataGenerator
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
    shear_range=0,
    zoom_range=0.1,
    # channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=None  # On pourrai peut être resize ici
    # data_format=K.image_data_format()
)


def ignore_file(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


class HierarchicalDatagen:
    def __init__(self, super_path, path_to_copy, nb_gen_by_folder, delete=True):
        self.nb_gen_by_folder = nb_gen_by_folder
        HierarchicalDatagen.copy_structure(super_path, path_to_copy, delete)
        db_files = HierarchicalDatagen.create_db(super_path)

        for folder, list_path in db_files.items():
            print(folder, "...")
            current_images = []
            for path in list_path:
                current_images.append(self.load_image(path))
            current_images = np.array(current_images)

            new_path = path.replace(super_path, path_to_copy)

            datagen.fit(current_images)
            # generator = datagen.flow(current_folder, save_to_dir=new_path,
            #                         batch_size=1, save_format='jpg')

            i = 1
            for batch in datagen.flow(current_images, batch_size=1):
                batch = np.asarray(batch)
                batch = np.reshape(batch, (150, 150))
                im = PIL.Image.fromarray(batch)
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(new_path + str(i) + ".jpg")

                if i == nb_gen_by_folder:
                    break
                i += 1

    def copy_structure(super_path, path_to_copy, delete):
        print("copying structure...")
        if delete:
            if os.path.exists(path_to_copy):
                shutil.rmtree(path_to_copy)
            shutil.copytree(super_path, path_to_copy, ignore=ignore_file)
        else:
            if not os.path.exists(path_to_copy):
                shutil.copytree(super_path, path_to_copy)

    def create_db(super_path):
        print("fetching all pictures...")
        files = glob.glob(super_path + '/**/*.jpg', recursive=True)
        db_files = dict()
        for path in files:
            folder = path.split(os.sep)[-2]
            value = db_files.get(folder)
            if value is None:
                db_files[folder] = [path]
            else:
                db_files.get(folder).append(path)
        return db_files

    def load_image(self, path_file, height=150, width=150):
        img = PIL.Image.open(path_file)
        np_img = np.array(img.copy())
        h, w = np_img.shape

        half_w = w // 2
        half_h = h // 2
        half_height = height // 2
        half_width = width // 2

        if w > width:
            if h > height:
                if h / height > w / width:
                    w_compensated = h * width // height
                    final_img = np.ones((h, w_compensated),
                                        dtype="uint8") * 255
                    final_img[0:h, int(w_compensated / 2 - w / 2)
                                       :int(w_compensated / 2 + w / 2)] = np_img
                else:
                    h_compensated = w * height // width
                    final_img = np.ones((h_compensated, w),
                                        dtype="uint8") * 255
                    final_img[int(h_compensated / 2 - h / 2)
                                  :int(h_compensated / 2 + h / 2), 0:w] = np_img
            else:
                h_compensated = w * height // width
                final_img = np.ones((h_compensated, w), dtype="uint8") * 255
                final_img[int(h_compensated / 2 - h / 2)                          :int(h_compensated / 2 + h / 2), 0:w] = np_img
        else:
            if h > height:
                w_compensated = h * width // height
                final_img = np.ones((h, w_compensated), dtype="uint8") * 255
                final_img[0:h, int(w_compensated / 2 - w / 2)
                                   :int(w_compensated / 2 + w / 2)] = np_img
            else:
                final_img = np.ones((height, width), dtype="uint8") * 255
                final_img[int(height / 2 - h / 2):int(height / 2 + h / 2),
                          int(width / 2 - w / 2):int(width / 2 + w / 2)] = np_img

        if final_img.shape != (height, width):
            final_img = np.resize(final_img, (height, width))
        return np.reshape(final_img, (height, width, 1))


path1 = "E:\Polytech_Projects\pfe_plankton_classif\Dataset\DATASET\level0_new_hierarchique22"
path2 = "E:\Polytech_Projects\pfe_plankton_classif\Dataset\DATASET\level0_new_hierarchique22_datagen"
hier_datagen = HierarchicalDatagen(path1, path2, 2000)
