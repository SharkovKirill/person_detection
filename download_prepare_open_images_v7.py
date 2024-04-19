import fiftyone as fo
import fiftyone.zoo as foz
import pandas as pd
import os
import shutil
import yaml
from utils.only_people_from_yolo import list_files_in_directory

ID_CLASS_PERSON_NEW = 0
SUBSETS_TO_PREPARE = ["train", "validation", "test"]
SUBSETS_TO_RENAME = ["train", "validation", "test"]
MAX_SAMPLES = 50  # 10000000000
IMAGES_TYPE = ".jpg"


def download_fiftyone_open_images_v7():
    fo.config.dataset_zoo_dir = "./"

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        label_types=["detections"],  # all splits
        classes=["Man", "Woman", "Person"],
        only_matching=True,
        max_samples=MAX_SAMPLES,
    )


def prepare_fiftyone_open_images_v7(
    ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
):
    for subset in SUBSETS_TO_PREPARE:
        directory_subset_images = f"./open-images-v7/{subset}/data"
        base_txt_path = f"./open-images-v7/{subset}/labels/"
        df = pd.read_csv(f"./open-images-v7/{subset}/labels/detections.csv")
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
            txt_path = base_txt_path + file_name + ".txt"
            with open(txt_path, "w") as file:
                file.write("\n".join(grouped[file_name]))
    os.remove("./open-images-v7/info.json")
    for subset in SUBSETS_TO_RENAME:
        shutil.move(f"./open-images-v7/{subset}/data", f"./open-images-v7/{subset}/images")
        shutil.rmtree(f"./open-images-v7/{subset}/metadata")
        os.remove(f"./open-images-v7/{subset}/labels/detections.csv")
    shutil.move(f"./open-images-v7/validation", f"./open-images-v7/valid")
    data = {
        "names": ["person"],
        "nc": 1,
        "train": "open-images-v7/train/images",
        "val": "open-images-v7/valid/images",
        "test": "open-images-v7/test/images",
    }
    with open("./open-images-v7/data.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def download_prepare_openimagesv7(
    ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
):
    print("open-images-v7 download started")
    download_fiftyone_open_images_v7()
    print("open-images-v7 download finished")
    print("open-images-v7 preparing started")
    prepare_fiftyone_open_images_v7(
        ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
    )
    print("open-images-v7 preparing finished")


download_prepare_openimagesv7(
    ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
)
