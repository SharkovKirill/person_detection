import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

dir_path_train = "./Pascal VOC 2012.v1-raw.yolov8_prepared/train/"
dir_path_valid = "./Pascal VOC 2012.v1-raw.yolov8_prepared/valid/"


def show_bb_yolo(
    dataset_name: str,
    dir_path_for_im_and_labels: str,
    cols: int = 5,
    rows: int = 2,
    shuffle: bool = True,
    figsize=(40, 20)
):
    n_pictures_to_show = rows * cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    labels_path = os.listdir(dir_path_for_im_and_labels + "labels/")
    if shuffle:
        random.shuffle(labels_path)
    for i, file_name_txt in enumerate(labels_path[:n_pictures_to_show]):
        txt_path = dir_path_for_im_and_labels + f"labels/{file_name_txt}"
        img_path = (
            dir_path_for_im_and_labels
            + f'images/{file_name_txt.rstrip(".txt") + ".jpg"}'
        )
        print(i + 1, txt_path)
        with open(txt_path, "r") as txt_file:
            txt_info = txt_file.read().split("\n")
        image = plt.imread(img_path)
        height, width, channels = image.shape  # height - высота, width - ширина

        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()

        for bb in txt_info:
            class_id, x_relative, y_relative, w_relative, h_relative = list(
                map(float, bb.split(" "))
            )
            x_center = int(x_relative * width)
            y_center = int(y_relative * height)
            w = int(w_relative * width)
            h = int(h_relative * height)
            x = x_center - w / 2
            y = y_center - h / 2
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="g", facecolor="none"
            )
            ax.ravel()[i].add_patch(rect)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"bbox_images_examples/{dataset_name}_example_{n_pictures_to_show}.jpg")


show_bb_yolo('VOC', dir_path_train, cols=5, rows=2, shuffle=True, figsize=(40, 20))
