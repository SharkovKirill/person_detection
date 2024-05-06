import argparse
import sys
import os
import shutil
import yaml

from typing import List, Tuple, Dict, Any

from aug_datasets import save_augmented_copy_with_options, all_aug_params
from segmentation import initialize_sam, segment_one_image_with_options

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


def stage_apply_fisheye_aug(
    txt_paths: List[str], image_paths: List[str], fisheye_params: Dict[str, Any]
) -> List[Tuple[List[List[float]], ndarray]]:
    assert len(txt_paths) == len(image_paths)

    txt_after_aug = []
    for i in range(0, len(txt_paths)):
        txt_after_aug.append(
            save_augmented_copy_with_options(
                image_paths[i],
                image_paths[i],
                txt_paths[i],
                all_aug_params=fisheye_params,
                also_return_image=True,
            )
        )

    return txt_after_aug


def stage_apply_sam(
    txt_paths: List[str],
    image_paths: List[str],
    sam: Sam,
    after_fisheye_aug: List[Tuple[List[List[float]], ndarray]],
) -> None:
    assert len(txt_paths) == len(image_paths)

    for i in range(0, len(txt_paths)):
        correct_lines, image_rgb = after_fisheye_aug[i]

        segment_one_image_with_options(
            ID_CLASS_PERSON_NEW=0,
            one_image_path=image_paths[i],
            one_label_path=None,
            one_label_seg_path=txt_paths[i],
            sam=sam,
            image_rgb=image_rgb,
            correct_lines_float=correct_lines,
        )


def apply_stages(
    sampled_datasets: Dict[str, Dict[str, Any]],
    fisheye_params: Dict[str, Any],
    sam_params: Dict[str, str],
) -> None:
    aug_options = all_aug_params.copy()
    for param_name, param_config in fisheye_params.items():
        aug_options[param_name].update(param_config)

    sam = initialize_sam(sam_params["weights_path"], sam_params["model_type"])

    after_fisheye: List[Tuple[List[List[float]], ndarray]] = []
    for dataset_config in sampled_datasets.values():
        if dataset_config["apply_fisheye"]:
            after_fisheye = stage_apply_fisheye_aug(
                dataset_config["txt_paths"],
                dataset_config["image_paths"],
                aug_options,
            )
        if dataset_config["apply_sam"]:
            stage_apply_sam(
                dataset_config["txt_paths"],
                dataset_config["image_paths"],
                sam,
                after_fisheye,
            )


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

    sampled_dict = sample_datasets(
        yaml_data["datasets"], args.datasets_dir_path, args.output_dir_path
    )
    with open(
        os.path.join(os.getcwd(), args.output_dir_path, yaml_data["sampled_yaml_file"]),
        "w",
    ) as stream:
        config = {
            "train": [
                f"../{key}/train/images"
                for key, value in sampled_dict.items()
                if "train" in value["splits"]
            ],
            "test": [
                f"../{key}/test/images"
                for key, value in sampled_dict.items()
                if "test" in value["splits"]
            ],
            "val": [
                f"../{key}/valid/images"
                for key, value in sampled_dict.items()
                if "valid" in value["splits"]
            ],
            "nc": 1,
            "names": ["person"],
        }
        yaml.safe_dump(config, stream)

    apply_stages(sampled_dict, yaml_data["fisheye_params"], yaml_data["sam_params"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
