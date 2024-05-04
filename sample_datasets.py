import argparse
import sys
import os
import random
import shutil

from typing import List, Tuple

from aug_datasets import save_augmented_copy_with_options, aug_params
from segmentation import initialize_sam, segment_one_image_with_options

from numpy import ndarray

def create_dirtree_without_files(src, dst):
    src = os.path.abspath(src)
    src_prefix = len(src) + len(os.path.sep)

    os.makedirs(dst)

    for root, dirs, _ in os.walk(src):
        for dirname in dirs:
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
            os.mkdir(dirpath)


def stage_apply_fisheye_aug(
    txt_paths: List[str],
    image_paths: List[str],
    distortion_limits: Tuple[float, float],
    shift_limits: Tuple[float, float],
) -> List[Tuple[List[List[float]], ndarray]]:
    assert len(txt_paths) == len(image_paths)

    aug_options = aug_params
    aug_options["distort_limit"] = distortion_limits
    aug_options["shift_limit"] = shift_limits

    txt_after_aug = []
    for i in range(0, len(txt_paths)):
        txt_after_aug.append(
            save_augmented_copy_with_options(
                image_paths[i],
                image_paths[i],
                txt_paths[i],
                aug_params=aug_options,
                also_return_image=True,
            )
        )

    return txt_after_aug


def stage_apply_sam(
    txt_paths: List[str],
    image_paths: List[str],
    sam_weights_path: str,
    sam_model_type: str,
    after_fisheye_aug: List[Tuple[List[List[float]], ndarray]]
) -> None:
    assert len(txt_paths) == len(image_paths)

    sam = initialize_sam(sam_weights_path, sam_model_type)
    
    for i in range(0, len(txt_paths)):
        correct_lines, image_rgb = after_fisheye_aug[i]

        segment_one_image_with_options(
            ID_CLASS_PERSON_NEW=0,
            one_image_path=image_paths[i],
            one_label_path=None,
            one_label_seg_path=txt_paths[i],
            sam=sam,
            image_rgb=image_rgb,
            correct_lines_float=correct_lines
        )


def sample_and_copy_file(
    datasets_dir: str, output_dir: str, samples_number: int
) -> Tuple[List[str], List[str]]:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    create_dirtree_without_files(datasets_dir, output_dir)

    cwd = os.path.join(os.getcwd(), datasets_dir)
    datasets = [x for x in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, x))]

    txt_paths = []
    image_paths = []

    for dataset in datasets:
        dataset_dir = os.path.join(cwd, dataset)
        splits = [
            os.path.relpath(os.path.join(dataset_dir, x), cwd)
            for x in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, x))
        ]

        for split in splits:
            labels_dir = os.path.join(cwd, split, "labels")
            labels_dir_size = len(next(os.walk(labels_dir))[2])
            images_dir = os.path.join(cwd, split, "images")
            cur_samples_number = min(samples_number, labels_dir_size)

            txt_random_sample = random.choices(
                os.listdir(labels_dir), k=cur_samples_number
            )
            sample_name = [x.rstrip(".txt") for x in txt_random_sample]
            txt_random_sample = [
                os.path.relpath(os.path.join(split, "labels", x))
                for x in txt_random_sample
            ]

            deduct_img_type = next(os.walk(images_dir))[2][0]
            _, img_type_random = os.path.splitext(deduct_img_type)
            sample_name = [
                os.path.relpath(os.path.join(split, "images", x + img_type_random))
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

    return (txt_paths, image_paths)


def initialize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A script for sampling datasets for YOLO8 model training"
    )

    parser.add_argument(
        "-d",
        "--datasets-dir-path",
        required=True,
        dest="datasets_dir_path",
        help="A path to the directory with datasets",
    )
    parser.add_argument(
        "-n",
        "--samples-number",
        type=int,
        required=True,
        dest="samples_number",
        help="The number of samples from all datasets",
    )
    parser.add_argument(
        "-w",
        "--sam-weights-path",
        required=True,
        dest="sam_weights_path",
        help="A path to the weights for SAM model",
    )
    parser.add_argument(
        "-m",
        "--sam-model-type",
        required=True,
        dest="sam_model_type",
        choices=["vit_h", "vit_b", "vit_l"],
        help="A model type for SAM model",
    )

    parser.add_argument(
        "-o",
        "--output-dir-path",
        default="sampled_datasets",
        dest="output_dir_path",
        help="A path to the directory with the prepared samples",
    )
    parser.add_argument(
        "-s", "--random-seed", default=42, dest="random_seed", help="Random seed"
    )

    parser.add_argument(
        "-t",
        "--distort-limit",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        dest="distort_limit",
        help="Distortion limit for fisheye augmentaion",
    )
    parser.add_argument(
        "-f",
        "--shift-limit",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        dest="shift_limit",
        help="Shift limit for fisheye augmentaion",
    )

    return parser


def main() -> int:
    parser = initialize_parser()
    args = parser.parse_args()

    random.seed(a=args.random_seed)
    txt_paths, image_paths = sample_and_copy_file(
        args.datasets_dir_path, args.output_dir_path, args.samples_number
    )

    after_fisheye = stage_apply_fisheye_aug(
        txt_paths,
        image_paths,
        (args.distort_limit[0], args.distort_limit[1]),
        (args.shift_limit[0], args.shift_limit[1]),
    )
    stage_apply_sam(txt_paths, image_paths, args.sam_weights_path, args.sam_model_type, after_fisheye)

    return 0


if __name__ == "__main__":
    sys.exit(main())
