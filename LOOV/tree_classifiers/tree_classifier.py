from anytree import Node, RenderTree
from anytree.search import findall
import os
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input


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

def create_tree(start):
    tree = dict()
    super_dir = start.split("/")[-1]
    tree[super_dir] = Node(start)

    for root, dirs, files in os.walk(start):
        if root == start:
            continue

        path = root.split("/")
        tree[path[-1]] = Node(path[-1], parent=tree[path[-2]])

    return tree[super_dir]


def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.name))

def tttt():
    tree = create_tree("eco_taxa_codebase")
    print_tree(tree)

def create_model(nb_classes):
    model = Sequential()
    model.add(Flatten(input_shape=(150, 150, 1)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model


def train_model(model, data_dir):
    train_generator = datagen.flow_from_directory(
        data_dir, batch_size=8, target_size=(150, 150), color_mode='grayscale')

    model.fit_generator(train_generator, steps_per_epoch=120,
                        #steps_per_epoch= nb_img //b batch_size,
                        validation_data=train_generator, validation_steps= 20,
                        #validation_steps= nb_img // batch_size,
                        epochs=3, workers=4)

    save_model(model, "models/model_" + data_dir + ".h5")


def get_directories(super_path):
    tree = create_tree(super_path)
    print_tree(tree)

    nodes = []

    for pre, fill, node in RenderTree(tree):
        path = [str(_.name) for _ in node.path]
        print(path)
        if len(path) > 1:
            nodes.append(path[-2])

    nodes = list(set(nodes))
    
    path_nodes = []
    for node in nodes:
        nn = findall(tree, lambda n: node == n.name)[0]
        path = "/".join([str(node.name) for node in nn.path])
        path_nodes.append(path)

    return path_nodes


## Take Care !! 
## Folders should have differents names !
## MANDATORY
def multi_classifier():
    super_path = "/home/user/Project/pfe_plankton_classif/Dataset/super_dataset"
    directories = get_directories(super_path)
    ddd = directories[0]
    files = [_ for _ in os.listdir(ddd)]
    print(files)

    
multi_classifier()