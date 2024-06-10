import json
import os
from PIL import Image
from typing import List
import shutil
import argparse
import sys


def get_coco_from_yolo_full_datasets(
    datasets_dir_path: List[str],
    output_dir_path: str,
    copy_images=False,
):
    os.makedirs(os.path.join(output_dir_path, "annotations"), exist_ok=True)
    if copy_images:
        os.makedirs(os.path.join(output_dir_path, "images"), exist_ok=True)
    id_img = 1
    categories = [{"id": 1, "name": "person"}]
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    for datasets_path in datasets_dir_path:
        possible_splits = ["train", "test", "valid"]
        dirs_splits = []

        for root, dirs, files in os.walk(datasets_path):
            for dir_p in dirs:
                if dir_p in possible_splits:
                    dirs_splits.append(os.path.join(root, dir_p))

        for input_dir in dirs_splits:
            print(input_dir)
            images_path = os.path.join(input_dir, "images")
            labels_path = os.path.join(input_dir, "labels")

            for image_file_name in os.listdir(images_path):
                image_path = os.path.join(images_path, image_file_name)

                if "aug" in os.path.normpath(image_path).split(os.sep):
                    new_image_file_name = "aug_" + image_file_name
                else:
                    new_image_file_name = image_file_name

                if copy_images:
                    shutil.copyfile(
                        image_path,
                        os.path.join(output_dir_path, "images", new_image_file_name),
                    )
                image = Image.open(image_path)
                width, height = image.size
                image_dict = {
                    "id": id_img,
                    "width": width,
                    "height": height,
                    "file_name": new_image_file_name,
                }

                coco_dataset["images"].append(image_dict)

                with open(
                    os.path.join(
                        labels_path, f'{".".join(image_file_name.split(".")[:-1])}.txt'
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
                        "category_id": 1,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                    }
                    coco_dataset["annotations"].append(ann_dict)
                id_img += 1

    with open(
        os.path.join(output_dir_path, "annotations", "instances_default.json"), "w"
    ) as f:
        json.dump(coco_dataset, f)


def initialize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A script for translating lables from yolo to coco"
    )

    parser.add_argument(
        "-d",
        "--datasets-dir-paths",
        required=True,
        dest="datasets_dir_path",
        help="A path to the directory with datasets",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--output-dir-path",
        default="./sampled_datasets_coco_labels",
        dest="output_dir_path",
        help="A path to the directory with the prepared samples",
        required=True,
    )

    parser.add_argument("--copyimages", action="store_true")

    return parser


def main() -> int:
    parser = initialize_parser()
    args = parser.parse_args()
    get_coco_from_yolo_full_datasets(
        datasets_dir_path=["./" + dataset for dataset in args.datasets_dir_path],
        output_dir_path="./" + args.output_dir_path,
        copy_images=args.copyimages,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# python sample_yolo_to_coco.py -d sampled_datasets_1 -o sampled_datasets_coco_labels --copyimages
