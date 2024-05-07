import albumentations as A
import matplotlib.pyplot as plt
import cv2
from typing import List
from numpy import ndarray

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
    "perspective_params": {"scale": (0.1, 0.15), "p": 0.5, "keep_size": True},
}


def aug_bboxes_and_image(
    bboxes_list_per_image: List[List[float]],  # List[class_id, x_c, y_c, w, h]
    image: ndarray,
    all_aug_params: dict = all_aug_params,
):
    ID_CLASS_PERSON_NEW = int(bboxes_list_per_image[0][0])
    default_boxes_prepared: List[List[float]] = []  # list of lists
    for box_list in bboxes_list_per_image:
        default_boxes_prepared.append(
            [box_list[1], box_list[2], box_list[3], box_list[4], ID_CLASS_PERSON_NEW]
        )  # changing the order for augmentation

    transform = A.Compose(
        [
            A.OpticalDistortion(**all_aug_params["distortion_params"]),
            A.HorizontalFlip(**all_aug_params["horizontal_params"]),
            A.Perspective(**all_aug_params["perspective_params"]),
        ],
        bbox_params=A.BboxParams(format="yolo"),
    )
    transformed = transform(image=image, bboxes=default_boxes_prepared)
    transformed_image = transformed["image"]
    transformed_bboxes = transformed["bboxes"]  # wrong order
    correct_lines_float = []  # correct order like List[List[x_center, y_center, w, h]]
    if len(transformed_bboxes)>=1:
        for one_annotation in transformed_bboxes:
            x_center, y_center, w, h = (
                one_annotation[0],
                one_annotation[1],
                one_annotation[2],
                one_annotation[3],
            )
            correct_lines_float.append([ID_CLASS_PERSON_NEW, x_center, y_center, w, h])

    return (
        correct_lines_float,
        transformed_image,
    )  # List[List[x_center, y_center, w, h]], np.ndarray
