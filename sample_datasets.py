import argparse
import sys
import os
import shutil
import yaml
import time
from typing import List, Dict, Any

from aug_datasets import aug_bboxes_and_image, all_aug_params
from segmentation import initialize_sam, segment_one_image
from utils.vizualize_bboxes import (
    read_bboxes,
    read_image,
    save_image,
    save_bboxes,
)
from segment_anything.modeling import Sam

from numpy import ndarray
from numpy import random as rd


def create_dirtree_without_files(src: str, dst: str):
    src = os.path.abspath(src)
    src_prefix = len(src) + len(os.path.sep)

    os.makedirs(dst)

    for root, dirs, _ in os.walk(src):
        for dirname in dirs:
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
            os.mkdir(dirpath)


def apply_stages(
    sampled_datasets: Dict[str, Dict[str, Any]],
    fisheye_params: Dict[str, Any],
    sam_params: Dict[str, str],
) -> None:
    aug_options = all_aug_params.copy()
    for param_name, param_config in fisheye_params.items():
        aug_options[param_name].update(param_config)

    sam = initialize_sam(sam_params["weights_path"], sam_params["model_type"])

    for dataset_config in sampled_datasets.values():
        if dataset_config["apply_fisheye"] or dataset_config["apply_sam"]:
            assert len(dataset_config["txt_paths"]) == len(
                dataset_config["image_paths"]
            )
            time_before_aug_sam = time.time()
            for i in range(0, len(dataset_config["txt_paths"])):
                bboxes_per_image = read_bboxes(dataset_config["txt_paths"][i])
                one_image = read_image(dataset_config["image_paths"][i])
                
                if len(bboxes_per_image)==0:
                    continue
                
                if dataset_config["apply_fisheye"]:
                    bboxes_per_image, one_image = aug_bboxes_and_image(
                        bboxes_per_image,
                        one_image,
                        aug_options,
                    )

                if len(bboxes_per_image)==0:
                    continue
                
                if dataset_config["apply_sam"]:
                    bboxes_per_image, one_image = segment_one_image(
                        sam, one_image, bboxes_per_image
                    )

                if len(bboxes_per_image)==0:
                    continue
                
                save_bboxes(bboxes_per_image, dataset_config["txt_paths"][i])
                save_image(one_image, dataset_config["image_paths"][i])
            print(f'Augmentation or SAM for {len(dataset_config["image_paths"])} paths took {time.time() - time_before_aug_sam}s')


def sample_datasets(
    datasets: List[Dict[str, Any]],
    datasets_dir: str,
    output_dir: str,
) -> Dict[str, Dict[str, Any]]:
    cwd = os.path.join(os.getcwd(), datasets_dir)

    res = {}
    for _ in datasets:
        for dataset_name, dataset_config in _.items():
            if not os.path.exists(os.path.join(cwd, dataset_name)):
                continue

            txt_paths = []
            image_paths = []
            samples_number = dataset_config.get("samples_number", -1)

            for split_name, split_value in dataset_config["splits"].items():
                labels_dir = os.path.join(cwd, dataset_name, split_name, "labels")
                labels_dir_size = len(next(os.walk(labels_dir))[2])
                images_dir = os.path.join(cwd, dataset_name, split_name, "images")

                if samples_number == -1:
                    config_value = split_value
                else:
                    config_value = int(round(samples_number * split_value))
                cur_samples_number = min(config_value, labels_dir_size)

                txt_random_sample = rd.choice(
                    os.listdir(labels_dir), size=cur_samples_number, replace=False
                )
                sample_name = [x.rstrip(".txt") for x in txt_random_sample]
                txt_random_sample = [
                    os.path.join(dataset_name, split_name, "labels", x)
                    for x in txt_random_sample
                ]

                deduct_img_type = next(os.walk(images_dir))[2][0]
                _, img_type_random = os.path.splitext(deduct_img_type)
                sample_name = [
                    os.path.join(
                        dataset_name, split_name, "images", x + img_type_random
                    )
                    for x in sample_name
                ]

                for txt_file in txt_random_sample:
                    output_dir_path = os.path.join(output_dir, txt_file)
                    txt_paths.append(output_dir_path)
                    shutil.copyfile(os.path.join(cwd, txt_file), output_dir_path)
                for img_file in sample_name:
                    output_dir_path = os.path.join(output_dir, img_file)
                    image_paths.append(output_dir_path)
                    shutil.copyfile(os.path.join(cwd, img_file), output_dir_path)

            res[dataset_name] = {
                "image_paths": image_paths,
                "txt_paths": txt_paths,
                "apply_sam": "sam" in dataset_config["stages"],
                "apply_fisheye": "fisheye" in dataset_config["stages"],
                "splits": dataset_config["splits"].keys(),
            }

    return res


def initialize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A script for sampling datasets for YOLO8 model training"
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
        help="A path to the directory with the prepared samples",
    )

    return parser


def main() -> int:
    parser = initialize_parser()
    args = parser.parse_args()

    yaml_data = {}
    with open(os.path.join(os.getcwd(), args.config_path), "r") as stream:
        yaml_data = yaml.safe_load(stream)

    rd.seed(yaml_data.get("random_seed", None))
    if os.path.exists(args.output_dir_path):
        shutil.rmtree(args.output_dir_path)
    create_dirtree_without_files(args.datasets_dir_path, args.output_dir_path)
    time_before_sample_datasets = time.time()
    sampled_dict = sample_datasets(
        yaml_data["datasets"], args.datasets_dir_path, args.output_dir_path
    )
    with open(
        os.path.join(os.getcwd(), args.output_dir_path, yaml_data["sampled_yaml_file"]),
        "w",
    ) as stream:
        config = {
            "train": [
                f"./{key}/train/images"
                for key, value in sampled_dict.items()
                if "train" in value["splits"]
            ],
            "test": [
                f"./{key}/test/images"
                for key, value in sampled_dict.items()
                if "test" in value["splits"]
            ],
            "val": [
                f"./{key}/valid/images"
                for key, value in sampled_dict.items()
                if "valid" in value["splits"]
            ],
            "nc": 1,
            "names": ["person"],
            "path": os.path.join(os.getcwd(), args.output_dir_path),
            #"path": f"../{args.output_dir_path}",
        }
        yaml.safe_dump(config, stream)
    time_before_apply_stages = time.time()
    print('time for copy:', time_before_apply_stages - time_before_sample_datasets)
    apply_stages(sampled_dict, yaml_data["fisheye_params"], yaml_data["sam_params"])
    print('time for aug and SAM:', time.time() - time_before_apply_stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
