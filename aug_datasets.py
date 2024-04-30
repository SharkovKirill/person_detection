import albumentations as A
import matplotlib.pyplot as plt
import cv2

ID_CLASS_PERSON_NEW = 0
aug_params = {
    "distort_limit": (2.8, 2.8),
    "shift_limit": (10, 10),
    "border_mode": cv2.BORDER_CONSTANT,
    "p": 1,
}


def save_augmented_copy(
    ID_CLASS_PERSON_NEW: int = ID_CLASS_PERSON_NEW,
    aug_params: dict = aug_params,
    one_image_path: str = "./Kitti/train/images/000035.png", # example
    one_image_aug_path: str = "./000035_aug.png", # example
    one_label_path: str = "./Kitti/train/labels/000035.txt", # example
    one_label_aug_path: str = "./000035_aug.txt", # example
):

    with open(one_label_path, "r") as bb_file:
        boxes = bb_file.read().split("\n")
    default_boxes = []  # list of lists
    for one_box in boxes:
        box_list = list(map(float, one_box.split(" ")))
        default_boxes.append(
            [box_list[1], box_list[2], box_list[3], box_list[4], ID_CLASS_PERSON_NEW]
        )  # changing the order for augmentation
    default_image = plt.imread(one_image_path)
    print(default_boxes)
    transform = A.Compose(
        [A.OpticalDistortion(**aug_params)], bbox_params=A.BboxParams(format="yolo")
    )
    transformed = transform(image=default_image, bboxes=default_boxes)
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
    elif len(default_image.shape) == 2: # didn
        plt.imsave(one_image_aug_path, transformed_image, cmap="grey")


save_augmented_copy()
