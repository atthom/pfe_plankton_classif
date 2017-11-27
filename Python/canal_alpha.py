from PIL import Image
import numpy as np
import os


def generate_pictures():
     # replace ("pfe_plankton_classif/LOOV/uvp5ccelter_group1/", "YOLO/Generate/Alpha_uvp5/")

    train_data_dir = "/home/mahiout/Documents/Projets/YOLO/uvp5ccelter_group1/"

    dir_names = [_ for _ in os.listdir(train_data_dir)]
    print("Generating pictures...")
    for dirname in dir_names:
        print(dirname)
        files = [_ for _ in os.listdir(train_data_dir + dirname)]
        lenfiles = len(files)

        i = 0
        for fi in files:
            i += 1
            print("Picture n°", i, "sur", lenfiles, "...")
            path = train_data_dir + dirname + "/" + fi

            new_path = create_dir_if_needed(path)
            new_path = new_path.replace("jpg", "png")

            img = Image.open(new_path).convert("RGBA")
            new_img = imgWithAlpha(img)
            new_img.save(path, 'png')



def create_dir_if_needed(path):
    dd = path.replace("uvp5ccelter_group1/", "/Generate/Alpha_uvp5/")
    dd = dd.split("/")[0:len(dd.split("/")) - 1]
    dd = "/".join(dd)

    if not os.path.isdir(dd):
        os.makedirs(dd)
    return dd




def imgWithAlpha(img):
    np_img = np.array(img.copy())

    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            val = np_img[i,j,0]
            print(val)
            if val > 240:
                np_img[i,j,3] = 0
    return Image.fromarray(np_img, "RGBA")

def main():
    new = "/home/mahiout/Documents/Projets/YOLO/Generate/Alpha_uvp5/"
    img = Image.open("952.jpg").convert("RGBA")

    img2 = imgWithAlpha(img)
    img2.save("952.png", 'png')



    # RGB (255,255,255) => on le met en canal Alpha


main()
print("\n\ndone.")
