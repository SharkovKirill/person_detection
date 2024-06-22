import os
import pandas as pd

from utils.vizualize_bboxes import show_bb_yolo, save_hists

DATASETS = (
    ("Kitti", ".png"),
    ("Pascal-VOC-2012-1", ".jpg"),
    ("coco-2017", ".jpg"),
    ("WiderPerson", ".jpg"),
    ("Construction-Site-Safety-30", ".jpg"),
    ("open-images-v7", ".jpg"),
    ("Person-Detection-Fisheye-1", ".jpg")
)

DATASET_INFO: pd.DataFrame = pd.DataFrame(columns=["split", "total_pic", "total_people"])

for dataset in DATASETS:
    dataset_name, dataset_image_type = dataset
    dataset_dir = os.path.join(os.getcwd(), "datasets", dataset_name)
    if not os.path.exists(dataset_dir):
        continue

    splits = [ name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name)) ]
    for split in splits:
        show_bb_yolo(
            dataset_name,
            os.path.join(dataset_dir, split),
            cols=5,
            rows=2,
            shuffle=True,
            data_type=split,
            figsize=(40, 20),
            pictures_type=dataset_image_type,
        )

        to_append = save_hists(dataset_name, os.path.join(dataset_dir, split), split, figsize=(15, 6))
        DATASET_INFO = pd.concat([DATASET_INFO, to_append])

DATASET_INFO.to_csv("hists_datasets/dataset_info.csv")
