import albumentations as A
import matplotlib.pyplot as plt
import cv2
from typing import List

ID_CLASS_PERSON_NEW = 0

all_aug_params = {
    "distortion_params": {
        "distort_limit": (2.8, 2.8),
        "shift_limit": (10, 10),
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 0.5,
    },
    "horizontal_params": {
        "p": 0.5,
    },
    "perspective_params":{ 
        "scale":(0.1, 0.15),
        "p":0.5,
        "keep_size":True
    }
}


def save_augmented_copy(
    one_image_path: str,
    one_image_aug_path,
    one_label_path: str,
    one_label_aug_path: str,
    ID_CLASS_PERSON_NEW: int = ID_CLASS_PERSON_NEW,
    aug_params: dict = all_aug_params,
):

    with open(one_label_path, "r") as bb_file:
        boxes = bb_file.read().split("\n")
    default_boxes_prepared = []  # list of lists
    for one_box in boxes:
        box_list = list(map(float, one_box.split(" ")))
        default_boxes_prepared.append(
            [box_list[1], box_list[2], box_list[3], box_list[4], ID_CLASS_PERSON_NEW]
        )  # changing the order for augmentation

    default_image = cv2.imread(one_image_path)
    if len(default_image.shape) == 3:
        default_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2RGB)

    transform = A.Compose(
        [A.OpticalDistortion(**aug_params)], bbox_params=A.BboxParams(format="yolo")
    )
    transformed = transform(image=default_image, bboxes=default_boxes_prepared)
    transformed_image = transformed["image"]
    transformed_bboxes = transformed["bboxes"]  # wrong order
    correct_lines = []  # correct order
    for one_annotation in transformed_bboxes:
        x_center, y_center, w, h = (
            one_annotation[0],
            one_annotation[1],
            one_annotation[2],
            one_annotation[3],
        )
        new_correct_line = f"{ID_CLASS_PERSON_NEW} {x_center} {y_center} {w} {h}"
        correct_lines.append(new_correct_line)
    with open(one_label_aug_path, "w") as file:
        file.write("\n".join(correct_lines))
    if len(default_image.shape) == 3:
        plt.imsave(one_image_aug_path, transformed_image)
    elif len(default_image.shape) == 2:
        plt.imsave(one_image_aug_path, transformed_image, cmap="grey")


def save_augmented_copy_with_options(
    one_image_path: str,
    one_image_aug_path: str,
    one_label_path: str,
    one_label_aug_path: str = None,
    ID_CLASS_PERSON_NEW: int = ID_CLASS_PERSON_NEW,
    all_aug_params: dict = all_aug_params,
    also_return_image: bool = True,
    save_new_txt: bool = False,
):

    with open(one_label_path, "r") as bb_file:
        boxes = bb_file.read().split("\n")
    default_boxes_prepared = []  # list of lists
    for one_box in boxes:
        box_list = list(map(float, one_box.split(" ")))
        default_boxes_prepared.append(
            [box_list[1], box_list[2], box_list[3], box_list[4], ID_CLASS_PERSON_NEW]
        )  # changing the order for augmentation
    default_image = cv2.imread(one_image_path)
    if len(default_image.shape) == 3:
        default_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2RGB)
    transform = A.Compose(
        [
            A.OpticalDistortion(**all_aug_params["distortion_params"]),
            A.HorizontalFlip(**all_aug_params["horizontal_params"]),
            A.Perspective(**all_aug_params["perspective_params"])
        ],
        bbox_params=A.BboxParams(format="yolo"),
    )
    transformed = transform(image=default_image, bboxes=default_boxes_prepared)
    transformed_image = transformed["image"]
    transformed_bboxes = transformed["bboxes"]  # wrong order
    correct_lines_float = []  # correct order like List[List[x_center, y_center, w, h]]
    for one_annotation in transformed_bboxes:
        x_center, y_center, w, h = (
            one_annotation[0],
            one_annotation[1],
            one_annotation[2],
            one_annotation[3],
        )
        correct_lines_float.append([x_center, y_center, w, h])

    if len(default_image.shape) == 3:
        plt.imsave(one_image_aug_path, transformed_image)
    elif len(default_image.shape) == 2:
        plt.imsave(one_image_aug_path, transformed_image, cmap="grey")

    if save_new_txt:
        if one_label_aug_path is not None:
            correct_lines_str = [
                f"{ID_CLASS_PERSON_NEW} {list_of_bboxes[0]} {list_of_bboxes[1]} {list_of_bboxes[2]} {list_of_bboxes[3]}"
                for list_of_bboxes in correct_lines_float
            ]  # List[f"{ID_CLASS_PERSON_NEW} {x_center} {y_center} {w} {h}"]
            with open(one_label_aug_path, "w") as file:
                file.write("\n".join(correct_lines_str))
        else:
            print("The path to save annotations is not specified! TXT was NOT saved.")
    if also_return_image:
        return (
            correct_lines_float,
            transformed_image,
        )  # List[List[x_center, y_center, w, h]], np.ndarray
    elif also_return_image == False:
        return correct_lines_float  # List[List[x_center, y_center, w, h]]
