import yaml

from utils.only_people_from_yolo import (
    list_files_in_directory,
    has_person_in_txt,
    del_txt_not_in_list,
    del_images_not_in_list_txt,
    relabel_and_del_useless_classes_from_yolo8,
)

PERSON_ID = ("8",)
SPLITS = ["test", "train", "valid"]

def prepare_dataset() -> None:
    for split in SPLITS:
        cwd = f"./Construction-Site-Safety-30/{split}"
        labels = cwd + "/labels"
        images = cwd + "/images"
        
        all_annotations = list_files_in_directory(labels, set)

        has_persons_list = has_person_in_txt(PERSON_ID, all_annotations, labels, set)
        relabel_and_del_useless_classes_from_yolo8(PERSON_ID, 0, has_persons_list, labels)
        
        del_txt_not_in_list(labels, has_persons_list)
        del_images_not_in_list_txt(images, has_persons_list)
        
        data = {
            "names": ["Person"],
            "nc": 1,
            "train": "Construction-Site-Safety-30/train/images",
            "val": "Construction-Site-Safety-30/valid/images",
            "test": "Construction-Site-Safety-30/test/images",
        }
        with open("./Construction-Site-Safety-30/data.yaml", "w") as file:
            yaml.dump(data, file, default_flow_style=False)

def transform_construction_safety() -> None:
    prepare_dataset()
