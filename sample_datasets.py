import argparse
import os
import shutil
import sys
import time
from typing import Any, Dict, List

from tqdm import tqdm

import yaml
from numpy import random as rd

from aug_datasets import all_aug_params, aug_bboxes_and_image
from segmentation import initialize_sam, segment_one_image
from utils.vizualize_bboxes import read_bboxes, read_image, save_bboxes, save_image


def create_dirtree_without_files(src: str, dst: str):
    src = os.path.abspath(src)
    src_prefix = len(src) + len(os.path.sep)

    dirpaths = []
    for root, dirs, _ in os.walk(src):
        for dirname in dirs:
            dirpaths.append(os.path.join(dst, root[src_prefix:], dirname))

    os.makedirs(dst)
    for dirpath in dirpaths:
        os.mkdir(dirpath)


def apply_stages(
    sampled_datasets: Dict[str, Dict[str, Any]],
    fisheye_params: Dict[str, Any],
    sam_params: Dict[str, str],
    output_dir: str
) -> None:
    aug_options = all_aug_params.copy()
    for param_name, param_config in fisheye_params.items():
        aug_options[param_name].update(param_config)

    sam = initialize_sam(sam_params["weights_path"], sam_params["model_type"])

    for dataset_name, dataset_config in sampled_datasets.items():
        assert len(dataset_config["txt_paths"]) == len(dataset_config["image_paths"])

        pbar = tqdm(range(0, len(dataset_config["txt_paths"])))
        for i in pbar:
            pbar.set_description(f"{dataset_name}")
            pbar.set_postfix_str(dataset_config["txt_paths"][i])

            bboxes_per_image = read_bboxes(
                os.path.join(output_dir, dataset_config["txt_paths"][i])
            )
            one_image = read_image(
                os.path.join(output_dir, dataset_config["image_paths"][i])
            )

            if len(bboxes_per_image) == 0:
                continue

            if dataset_config["apply_fisheye"]:
                bboxes_per_image, one_image = aug_bboxes_and_image(
                    bboxes_per_image,
                    one_image,
                    aug_options,
                )

            if len(bboxes_per_image) == 0:
                continue

            if dataset_config["apply_sam"]:
                bboxes_per_image, one_image = segment_one_image(
                    sam, one_image, bboxes_per_image
                )

            if len(bboxes_per_image) == 0:
                continue

            output_aug_txt = os.path.join(output_dir, "aug", dataset_config["txt_paths"][i])
            output_aug_image = os.path.join(output_dir, "aug", dataset_config["image_paths"][i])

            save_bboxes(bboxes_per_image, output_aug_txt)
            save_image(one_image, output_aug_image)


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
                    txt_paths.append(txt_file)
                    shutil.copyfile(os.path.join(cwd, txt_file), os.path.join(output_dir, txt_file))
                for img_file in sample_name:
                    image_paths.append(img_file)
                    shutil.copyfile(os.path.join(cwd, img_file), os.path.join(output_dir, img_file))

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
    create_dirtree_without_files(
        args.output_dir_path, os.path.join(args.output_dir_path, "aug")
    )

    time_before_sample_datasets = time.time()

    sampled_dict = sample_datasets(
        yaml_data["datasets"], args.datasets_dir_path, args.output_dir_path
    )

    with open(
        os.path.join(os.getcwd(), args.output_dir_path, yaml_data["sampled_yaml_file"]),
        "w",
    ) as stream:
        def aug_exist(dataset: Dict[str, Dict[str, Any]]) -> bool:
            return "apply_sam" in dataset or "apply_fisheye" in dataset

        def split_exist(split: str, splits: Dict[str, Dict[str, Any]]) -> bool:
            return split in splits

        train = [
            f"./{key}/train/images"
            for key, value in sampled_dict.items()
            if split_exist("train", value["splits"])
        ]
        train.extend(
            [
                f"./aug/{key}/train/images"
                for key, value in sampled_dict.items()
                if split_exist("train", value["splits"]) and aug_exist(value)
            ]
        )

        test = [
            f"./{key}/test/images"
            for key, value in sampled_dict.items()
            if split_exist("train", value["splits"])
        ]
        test.extend(
            [
                f"./aug/{key}/test/images"
                for key, value in sampled_dict.items()
                if split_exist("test", value["splits"]) and aug_exist(value)
            ]
        )

        val = [
            f"./{key}/valid/images"
            for key, value in sampled_dict.items()
            if split_exist("valid", value["splits"])
        ]
        val.extend(
            [
                f"./aug/{key}/valid/images"
                for key, value in sampled_dict.items()
                if split_exist("valid", value["splits"]) and aug_exist(value)
            ]
        )

        config = {
            "train": train,
            "test": test,
            "val": val,
            "nc": 1,
            "names": ["person"],
            "path": os.path.join(os.getcwd(), args.output_dir_path),
            # "path": f"../{args.output_dir_path}", ## wsl
        }
        yaml.safe_dump(config, stream)

    time_before_apply_stages = time.time()
    print("time for copy:", time_before_apply_stages - time_before_sample_datasets)
    apply_stages(
        sampled_dict,
        yaml_data["fisheye_params"],
        yaml_data["sam_params"],
        args.output_dir_path,
    )
    print("time for aug and SAM:", time.time() - time_before_apply_stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
