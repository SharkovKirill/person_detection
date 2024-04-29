# !pip install fiftyone
# !pip install pycocotools

from pycocotools.coco import COCO
from typing import List

import os
import shutil
import yaml


ID_CLASS_PERSON_NEW = 0
IMAGES_TYPE = ".jpg"
SUBSETS_TO_PREPARE = ["train", "validation"]
SUBSETS_TO_RENAME = ["train", "validation", "test"]


def yolov8_from_coco(height, width, x, y, w_before, h_before):
    x_center = (x + 0.5 * w_before) / width
    y_center = (y + 0.5 * h_before) / height
    w = w_before / width
    h = h_before / height
    return x_center, y_center, w, h


def prepare_fiftyone_COCO(
    ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
):
    for subset in SUBSETS_TO_PREPARE:
        subset_path = f"./coco-2017/{subset}/"
        if not os.path.exists(subset_path + "labels"):
            os.mkdir(subset_path + "labels")
        coco = COCO(subset_path + "labels.json")
        for img_id in coco.getImgIds(catIds=[1]):
            img = coco.loadImgs(img_id)
            txt_name = img[0]["file_name"].rstrip(IMAGES_TYPE) + ".txt"
            txt_path = subset_path + "labels/" + txt_name
            height = img[0]["height"]
            width = img[0]["width"]
            correct_lines = []
            for ann_id in coco.getAnnIds(imgIds=[img_id], catIds=[1]):
                ann = coco.anns[ann_id]
                x_center, y_center, w, h = yolov8_from_coco(
                    height,
                    width,
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][2],
                    ann["bbox"][3],
                )
                new_correct_line = (
                    f"{ID_CLASS_PERSON_NEW} {x_center} {y_center} {w} {h}"
                )
                correct_lines.append(new_correct_line)
            with open(txt_path, "w") as file:
                file.write("\n".join(correct_lines))
    shutil.rmtree("./coco-2017/raw")
    os.remove("./coco-2017/info.json")
    for subset in SUBSETS_TO_RENAME:
        shutil.move(f"./coco-2017/{subset}/data", f"./coco-2017/{subset}/images")
        os.remove(f"./coco-2017/{subset}/labels.json")
    shutil.move(f"./coco-2017/validation", f"./coco-2017/valid")
    data = {
        "names": ["person"],
        "nc": 1,
        "train": "coco-2017/train/images",
        "val": "coco-2017/valid/images",
        "test": "coco-2017/test/images",
    }
    with open("./coco-2017/data.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def transform_coco(
    ID_CLASS_PERSON_NEW: int = ID_CLASS_PERSON_NEW,
    IMAGES_TYPE: str = IMAGES_TYPE,
    SUBSETS_TO_PREPARE: List[str] = SUBSETS_TO_PREPARE,
    SUBSETS_TO_RENAME: List[str] = SUBSETS_TO_RENAME
):
    print("fiftyone COCO preparing started")
    prepare_fiftyone_COCO(
        ID_CLASS_PERSON_NEW, IMAGES_TYPE, SUBSETS_TO_PREPARE, SUBSETS_TO_RENAME
    )
    print("fiftyone COCO preparing finished")
