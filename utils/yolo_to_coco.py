import json
import os
from PIL import Image
from typing import List
import shutil


def get_coco_from_yolo_full_datasets(
    datasets: List[str],
    new_dataset_dir: str,
    copy_images=False,
):
    if copy_images:
        os.makedirs(os.path.join(new_dataset_dir, "images"), exist_ok=True)
    id_img = 1
    categories = [{"id": 1, "name": "person"}]
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    for datasets_path in datasets_paths:
        dirs_splits = []

        for dir_p in os.listdir(datasets_path):
            dir_p1 = os.path.join(datasets_path, dir_p)
            if os.path.isdir(dir_p1):
                for dir_p2 in os.listdir(dir_p1):
                    dir_full = os.path.join(dir_p1, dir_p2)
                    if os.path.isdir(dir_full):
                        dirs_splits.append(dir_full)

        for input_dir in dirs_splits:
            print(input_dir)
            images_path = os.path.join(input_dir, "images")
            labels_path = os.path.join(input_dir, "labels")

            for image_file in os.listdir(images_path):
                image_path = os.path.join(images_path, image_file)
                if copy_images:
                    shutil.copyfile(
                        image_path, os.path.join(new_dataset_dir, "images", image_file)
                    )
                image = Image.open(image_path)
                width, height = image.size
                image_dict = {
                    "id": id_img,
                    "width": width,
                    "height": height,
                    "file_name": image_file,
                }

                coco_dataset["images"].append(image_dict)

                with open(
                    os.path.join(
                        labels_path, f'{".".join(image_file.split(".")[:-1])}.txt'
                    )
                ) as f:
                    annotations = f.readlines()

                for ann in annotations:
                    x, y, w, h = map(float, ann.strip().split()[1:])
                    x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
                    x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
                    ann_dict = {
                        "id": len(coco_dataset["annotations"]),
                        "image_id": id_img,
                        "category_id": 0,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                    }
                    coco_dataset["annotations"].append(ann_dict)
                id_img += 1

    with open(os.path.join(new_dataset_dir, "annotations.json"), "w") as f:
        json.dump(coco_dataset, f)


datasets_paths = ["./datasets", "./sampled_datasets_1"]
get_coco_from_yolo_full_datasets(
    datasets=datasets_paths, new_dataset_dir="./dataset_coco", copy_images=True
)
