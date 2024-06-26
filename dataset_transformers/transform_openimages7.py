from typing import List

import pandas as pd
import os
import shutil
import yaml
from utils.only_people_from_yolo import list_files_in_directory

ID_CLASS_PERSON_NEW = 0
SUBSETS_TO_PREPARE = ["train", "validation", "test"]
SUBSETS_TO_RENAME = ["train", "validation", "test"]
IMAGES_TYPE = ".jpg"


def prepare_fiftyone_open_images_v7(
    datasets_dir_path, ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
):
    cwd = os.path.join(datasets_dir_path, "open-images-v7")
    for subset in SUBSETS_TO_PREPARE:
        directory_subset_images = os.path.join(cwd, subset, "data")
        base_txt_path = os.path.join(cwd, subset, "labels")
        
        df = pd.read_csv(os.path.join(base_txt_path, "detections.csv"))
        df = df[["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]]
        df = df[
            (df["LabelName"] == "/m/01g317")
            | (df["LabelName"] == "/m/04yx4")
            | (df["LabelName"] == "/m/03bt1vf")
        ]
        df["x_center"] = (df["XMin"] + df["XMax"]) / 2
        df["y_center"] = (df["YMin"] + df["YMax"]) / 2
        df["w"] = df["XMax"] - df["XMin"]
        df["h"] = df["YMax"] - df["YMin"]
        file_names = list(
            map(
                lambda x: x.rstrip(IMAGES_TYPE),
                list_files_in_directory(directory_subset_images),
            )
        )
        df["full_line"] = df[["x_center", "y_center", "w", "h"]].apply(
            lambda row: str(ID_CLASS_PERSON_NEW)
            + " "
            + str(row["x_center"])
            + " "
            + str(row["y_center"])
            + " "
            + str(row["w"])
            + " "
            + str(row["h"]),
            axis=1,
        )
        grouped = df.groupby("ImageID")["full_line"].apply(list)
        for file_name in file_names:
            txt_path = os.path.join(base_txt_path, file_name + ".txt")
            with open(txt_path, "w") as file:
                file.write("\n".join(grouped[file_name]))

    os.remove(os.path.join(cwd, "info.json"))

    for subset in SUBSETS_TO_RENAME:
        shutil.move(os.path.join(cwd, subset, "data"), os.path.join(cwd, subset, "images"))
        shutil.rmtree(os.path.join(cwd, subset, "metadata"))
        os.remove(os.path.join(cwd, subset, "labels", "detections.csv"))

    shutil.move(os.path.join(cwd, "validation"), os.path.join(cwd, "valid"))
    data = {
        "names": ["person"],
        "nc": 1,
        "train": "open-images-v7/train/images",
        "val": "open-images-v7/valid/images",
        "test": "open-images-v7/test/images",
    }
    with open(os.path.join(cwd, "data.yaml"), "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def transform_openimages7(
    datasets_dir_path: str,
    ID_CLASS_PERSON_NEW: int = ID_CLASS_PERSON_NEW,
    IMAGES_TYPE: str = IMAGES_TYPE,
    SUBSETS_TO_PREPARE: List[str] = SUBSETS_TO_PREPARE,
    SUBSETS_TO_RENAME: List[str] = SUBSETS_TO_RENAME
):
    print("open-images-v7 preparing started")
    prepare_fiftyone_open_images_v7(
        datasets_dir_path, ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
    )
    print("open-images-v7 preparing finished")
