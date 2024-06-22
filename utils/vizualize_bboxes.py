import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from typing import List
import cv2
from numpy import ndarray
import pandas as pd


def str_box_to_int_floats(bbox: List[str]):
    class_id = [int(bbox[0])]
    bbox_params = list(map(float, bbox[1:]))
    bbox_params = list(map(lambda x: 1e-5 if x==0 else x, bbox_params))
    return class_id + bbox_params


def read_bboxes(one_label_path: str):
    with open(one_label_path, "r") as bb_file:
        bboxes = bb_file.read().split("\n")
    if len(bboxes[0]) > 0:
        bboxes_list = [str_box_to_int_floats(one_box.split(" ")) for one_box in bboxes]
    elif len(bboxes[0]) == 0:
        bboxes_list = []
    return bboxes_list


def read_image(one_image_path: str):
    image = cv2.imread(one_image_path)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_bboxes(bboxes, one_label_path):
    if len(bboxes) >= 1:
        correct_lines_str = "\n".join(
            [" ".join(list(map(str, one_box))) for one_box in bboxes]
        )  # List[f"{ID_CLASS_PERSON_NEW} {x_center} {y_center} {w} {h}"]
    else:
        correct_lines_str = ''
    with open(one_label_path, "w") as file:
        file.write(correct_lines_str)


def save_image(image: ndarray, one_image_path: str):
    if len(image.shape) == 3:
        plt.imsave(one_image_path, image)
    elif len(image.shape) == 2:
        plt.imsave(one_image_path, image, cmap="grey")


def show_bb_yolo(
    dataset_name: str,
    dir_path_for_im_and_labels: str,
    cols: int = 5,
    rows: int = 2,
    shuffle: bool = True,
    data_type: str = "train",
    figsize=(40, 20),
    pictures_type: str = ".jpg",
    show_images=False
):
    n_pictures_to_show = rows * cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    labels_path = os.listdir(os.path.join(dir_path_for_im_and_labels, "labels"))
    if shuffle:
        random.shuffle(labels_path)
    for i, file_name_txt in enumerate(labels_path[:n_pictures_to_show]):
        txt_path = os.path.join(dir_path_for_im_and_labels, "labels", file_name_txt)
        img_path = os.path.join(
            dir_path_for_im_and_labels,
            "images",
            file_name_txt.rstrip(".txt") + pictures_type,
        )
        print(i + 1, txt_path)
        with open(txt_path, "r") as txt_file:
            txt_info = txt_file.read().split("\n")
        image = plt.imread(img_path)
        if len(image.shape) == 3:
            height, width, channels = image.shape  # height - высота, width - ширина
            ax.ravel()[i].imshow(image)
        elif len(image.shape) == 2:
            height, width = image.shape
            ax.ravel()[i].imshow(image, cmap="grey")

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
    plt.savefig(
        f"bbox_images_examples/{dataset_name}_{data_type}_bboxes_example_{n_pictures_to_show}.jpg"
    )
    if show_images:
        plt.show()


def save_hists(
    dataset_name: str,
    dir_path_for_im_and_labels: str,
    data_type: str = "train",
    figsize=(40, 20),
    show_images=False
):
    labels_path = os.listdir(os.path.join(dir_path_for_im_and_labels, "labels"))
    n_people_on_image = []
    w_relative_list = []
    h_relative_list = []
    for i, file_name_txt in enumerate(labels_path):
        txt_path = os.path.join(dir_path_for_im_and_labels, "labels", file_name_txt)
        with open(txt_path, "r") as txt_file:
            txt_info = txt_file.read().split("\n")
            if len(txt_info) == 1:
                continue
            n_people_on_image.append(len(txt_info))
            for bb in txt_info:
                _, _, _, w_relative, h_relative = list(map(float, bb.split(" ")))
                w_relative_list.append(w_relative)
                h_relative_list.append(h_relative)
    print(sum(n_people_on_image) / len(n_people_on_image))
    
    to_append = pd.DataFrame(
        columns=["split", "total_pic", "total_people"],
    )
    to_append.loc[dataset_name] = [data_type, len(n_people_on_image), sum(n_people_on_image)]

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    figure.suptitle(f"{dataset_name} {data_type}")

    ax.ravel()[0].hist(n_people_on_image, bins=20)
    ax.ravel()[0].set_title(f"Среднее кол-во людей на картинке")

    ax.ravel()[1].hist(h_relative_list, bins=20)
    ax.ravel()[1].set_title(f"Средняя относительная высота человека")

    ax.ravel()[2].hist(w_relative_list, bins=20)
    ax.ravel()[2].set_title(f"Средняя относительная ширина человека")

    plt.savefig(f"hists_datasets/{dataset_name}_hist_count_{data_type}_.jpg")
    if show_images:
        plt.show()
    
    return to_append


def show_old_and_new_bb(
    one_image_old_path: str,
    one_image_new_path: str,
    one_label_old_path: str,
    one_label_new_path: str,
    figsize: tuple,
    title: str = "Before -> after",
):
    old_image = plt.imread(one_image_old_path)
    with open(one_label_old_path, "r") as bb_file:
        old_lines_boxes = bb_file.read().split("\n")
    old_bb_list = []  # list of lists
    for one_box in old_lines_boxes:
        one_box_list = list(map(float, one_box.split(" ")))
        old_bb_list.append(one_box_list)

    new_image = plt.imread(one_image_new_path)
    with open(one_label_new_path, "r") as bb_file:
        new_lines_boxes = bb_file.read().split("\n")
    new_bb_list = []  # list of lists
    for one_box in new_lines_boxes:
        one_box_list = list(map(float, one_box.split(" ")))
        new_bb_list.append(one_box_list)

    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax.ravel()[0].set_axis_off()
    ax.ravel()[1].set_axis_off()
    if len(old_image.shape) == 3:
        height, width, channels = old_image.shape  # height - высота, width - ширина
        ax.ravel()[0].imshow(old_image)
        ax.ravel()[1].imshow(new_image)
    elif len(old_image.shape) == 2:
        height, width = old_image.shape
        ax.ravel()[0].imshow(old_image, cmap="grey")
        ax.ravel()[1].imshow(new_image, cmap="grey")

    for i, bb_list in zip([0, 1], [old_bb_list, new_bb_list]):
        for bb in bb_list:
            _, x_relative, y_relative, w_relative, h_relative = bb
            x_center = int(x_relative * width)
            y_center = int(y_relative * height)
            w = int(w_relative * width)
            h = int(h_relative * height)
            x = x_center - w / 2
            y = y_center - h / 2
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=3, edgecolor="r", facecolor="none"
            )
            ax.ravel()[i].add_patch(rect)
    figure.suptitle(title)
    plt.show()
