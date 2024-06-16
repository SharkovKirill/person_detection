import yaml
import os
import argparse
import sys
import shutil
from typing import List, Dict
from utils.vizualize_bboxes import read_bboxes


def initialize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A script for filtering datasets for YOLO8 model training"
    )

    parser.add_argument(
        "-c",
        "--config-path",
        required=True,
        dest="config_path",
        help="A path to the yaml config file",
    )
    parser.add_argument(
        "-d",
        "--datasets-dir-path",
        required=True,
        dest="datasets_dir_path",
        help="A path to the directory with datasets",
    )
    parser.add_argument(
        "-o",
        "--output-dir-path",
        default="sampled_datasets",
        dest="output_dir_path",
        help="A path to the directory with the filtered samples",
    )

    return parser


def get_split_dirs(datasets: List[str], datasets_dir_path: str):
    possible_splits = [
        "train",
        "test",
        "valid",
        "val",
        "VisDrone2019-DET-train",
        "VisDrone2019-DET-test",
    ]
    dirs_splits_paths = []

    for root, dirs, files in os.walk(datasets_dir_path):
        for dir_p in dirs:
            if dir_p in possible_splits and any(
                elem in datasets for elem in os.path.normpath(root).split(os.sep)
            ):
                dirs_splits_paths.append(
                    os.path.join(*os.path.join(root, dir_p).split(os.sep)[1:])
                )
    return dirs_splits_paths


def create_dirs(
    dirs_splits_paths: List[str], output_dir_path: str, no_people_dir_name: str
):
    for split_dir in dirs_splits_paths:
        os.makedirs(os.path.join(output_dir_path, split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir_path, split_dir, "labels"), exist_ok=True)
        os.makedirs(
            os.path.join(output_dir_path, no_people_dir_name, split_dir, "images"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(output_dir_path, no_people_dir_name, split_dir, "labels"),
            exist_ok=True,
        )


def filter_copy_datasets(
    dirs_splits_paths: List[str],
    datasets_dir_path: str,
    output_dir_path: str,
    filter_params: Dict,
    no_people_dir_name: str,
):

    count_labels_no_people = 0
    dirs_no_people = []
    filter_type = filter_params["filter_type"]
    filter_width = filter_params["width"]
    filter_height = filter_params["height"]
    width_height_together = filter_params["width_height_together"]
    for split_path in dirs_splits_paths:
        print(f"Now: {split_path}")
        labels_old_path = os.path.join(datasets_dir_path, split_path, "labels")
        images_old_path = os.path.join(datasets_dir_path, split_path, "images")
        labels_new_path = os.path.join(output_dir_path, split_path, "labels")
        images_new_path = os.path.join(output_dir_path, split_path, "images")
        labels_new_empty_path = os.path.join(
            output_dir_path, no_people_dir_name, split_path, "labels"
        )
        images_new_empty_path = os.path.join(
            output_dir_path, no_people_dir_name, split_path, "images"
        )
        for image_file_name in os.listdir(images_old_path):
            label_file_name = f'{".".join(image_file_name.split(".")[:-1])}.txt'
            _, ext = os.path.splitext(label_file_name)
            try:  # if there are no labels for image
                bboxes = read_bboxes(os.path.join(labels_old_path, label_file_name))
            except:
                continue
            count_bboxes = len(bboxes)
            if count_bboxes == 0:
                count_labels_no_people += 1
                if split_path not in dirs_no_people:
                    dirs_no_people.append(split_path)
                shutil.copyfile(
                    os.path.join(labels_old_path, label_file_name),
                    os.path.join(labels_new_empty_path, label_file_name),
                )  # copy labels
                shutil.copyfile(
                    os.path.join(images_old_path, image_file_name),
                    os.path.join(images_new_empty_path, image_file_name),
                )  # copy image
            else:
                count_filtered_bboxes = 0
                for bbox in bboxes:
                    if width_height_together:
                        if (
                            filter_width[0] <= bbox[3] <= filter_width[1]
                            and filter_height[0] <= bbox[4] <= filter_height[1]
                        ):
                            count_filtered_bboxes += 1
                    elif not width_height_together:
                        if (
                            filter_width[0] <= bbox[3] <= filter_width[1]
                            or filter_height[0] <= bbox[4] <= filter_height[1]
                        ):
                            count_filtered_bboxes += 1
                if (filter_type == "AtLeastOne" and count_filtered_bboxes > 0) or (
                    filter_type == "Only" and count_filtered_bboxes == count_bboxes
                ):
                    shutil.copyfile(
                        os.path.join(labels_old_path, label_file_name),
                        os.path.join(labels_new_path, label_file_name),
                    )  # copy labels
                    shutil.copyfile(
                        os.path.join(images_old_path, image_file_name),
                        os.path.join(images_new_path, image_file_name),
                    )  # copy image
    print(f"Count lables with no people: {count_labels_no_people}", dirs_no_people)


def main() -> int:
    parser = initialize_parser()
    args = parser.parse_args()

    yaml_data = {}
    with open(os.path.join(os.getcwd(), args.config_path), "r") as stream:
        yaml_data = yaml.safe_load(stream)
    if yaml_data["clear_output_dir"]:
        if os.path.exists(args.output_dir_path):
            shutil.rmtree(args.output_dir_path)

    dirs_splits_paths = get_split_dirs(yaml_data["datasets"], args.datasets_dir_path)
    print(f"Dirs to be checked: {dirs_splits_paths}")
    create_dirs(
        dirs_splits_paths, args.output_dir_path, yaml_data["no_people_dir_name"]
    )
    filter_copy_datasets(
        dirs_splits_paths,
        args.datasets_dir_path,
        args.output_dir_path,
        yaml_data["filter_params"],
        yaml_data["no_people_dir_name"],
    )


if __name__ == "__main__":
    sys.exit(main())
# python sample_datasets_by_size.py -c configs/sampling_by_size/sampling_by_size_1.yaml -d sampled_datasets_example -o sample_datasets_example_filtered
