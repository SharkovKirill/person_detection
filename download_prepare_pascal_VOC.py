# !pip install roboflow
# !pip install python-dotenv

import os
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

from utils.only_people_from_yolo import (
    del_images_not_in_list_txt,
    del_txt_not_in_list,
    has_person_in_txt,
    list_files_in_directory,
    relabel_and_del_useless_classes_from_yolo8,
    load_local_env
)

ID_CLASSES_PERSON_BEFORE = ('14')
ID_CLASS_PERSON_NEW = 0


def download_yolo8_VOC(api_key):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("jacob-solawetz").project("pascal-voc-2012")
    dataset = project.version(1).download("yolov8")


def prepare_VOC(
    ID_CLASSES_PERSON_BEFORE,
    ID_CLASS_PERSON_NEW,
    directory_train_labels,
    directory_train_images,
    directory_valid_labels,
    directory_valid_images,
):
    file_names_train_labels = list_files_in_directory(directory_train_labels)
    file_names_valid_labels = list_files_in_directory(directory_valid_labels)

    file_names_with_person_train = has_person_in_txt(
        ID_CLASSES_PERSON_BEFORE, file_names_train_labels, directory_train_labels
    )
    file_names_with_person_valid = has_person_in_txt(
        ID_CLASSES_PERSON_BEFORE, file_names_valid_labels, directory_valid_labels
    )

    del_txt_not_in_list(directory_train_labels, file_names_with_person_train)
    del_txt_not_in_list(directory_valid_labels, file_names_with_person_valid)

    del_images_not_in_list_txt(directory_train_images, file_names_with_person_train, '.jpg')
    del_images_not_in_list_txt(directory_valid_images, file_names_with_person_valid, '.jpg')

    relabel_and_del_useless_classes_from_yolo8(
        ID_CLASSES_PERSON_BEFORE,
        ID_CLASS_PERSON_NEW,
        file_names_with_person_train,
        directory_train_labels,
    )
    relabel_and_del_useless_classes_from_yolo8(
        ID_CLASSES_PERSON_BEFORE,
        ID_CLASS_PERSON_NEW,
        file_names_with_person_valid,
        directory_valid_labels,
    )
    data = {
        'names': ['person'],
        'nc': 1,
        'train': 'Pascal-VOC-2012-1/train/images',
    }
    with open('./Pascal-VOC-2012-1/data.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def download_prepare(ID_CLASSES_PERSON_BEFORE, ID_CLASS_PERSON_NEW):
    load_local_env()
    API_KEY_ROBOFLOW = os.getenv("API_KEY_ROBOFLOW")
    print("PASCAL VOC download started")
    download_yolo8_VOC(API_KEY_ROBOFLOW)
    print("PASCAL VOC download finished")
    directory_train_labels = "./Pascal-VOC-2012-1/train/labels"
    directory_train_images = "./Pascal-VOC-2012-1/train/images"
    directory_valid_labels = "./Pascal-VOC-2012-1/valid/labels"
    directory_valid_images = "./Pascal-VOC-2012-1/valid/images"
    print("PASCAL VOC preparing started")
    prepare_VOC(
        ID_CLASSES_PERSON_BEFORE,
        ID_CLASS_PERSON_NEW,
        directory_train_labels,
        directory_train_images,
        directory_valid_labels,
        directory_valid_images,
    )
    print("PASCAL VOC preparing finished")


download_prepare(ID_CLASSES_PERSON_BEFORE, ID_CLASS_PERSON_NEW)
