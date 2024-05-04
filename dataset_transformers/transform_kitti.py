import matplotlib.pyplot as plt
import os
import shutil
import yaml

from utils.only_people_from_yolo import (
    list_files_in_directory,
    has_person_in_txt,
    del_txt_not_in_list,
    del_images_not_in_list_txt,
)

ID_CLASSES_PERSON_BEFORE = (
    "Pedestrian",
    "Cyclist",
)  # PersonSitting нет почему-то
ID_CLASS_PERSON_NEW = 0


def relabel_and_del_useless_classes_from_pytorch_KITTI(
    ID_CLASSES_PERSON_BEFORE: tuple,
    ID_CLASS_PERSON_NEW,
    file_names,
    directory_train_labels,
    directory_train_images,
):
    for file_name in file_names:
        txt_path = directory_train_labels + f"/{file_name}"
        img_path = directory_train_images + f'/{file_name.rstrip(".txt") + ".png"}'
        image = plt.imread(img_path)
        
        if len(image.shape) == 3:
            height, width, _ = image.shape 
        elif len(image.shape) == 2:
            height, width = image.shape
            
        correct_lines = []
        with open(txt_path, "r") as file:
            all_lines = file.read().split("\n")
            for line in all_lines:
                for particular_word in ID_CLASSES_PERSON_BEFORE:
                    if line.startswith(particular_word):
                        splitted = line.split(" ")
                        xleft = float(splitted[4])
                        yleft = float(splitted[5])
                        xright = float(splitted[6])
                        yright = float(splitted[7])
                        x_center = (xleft + xright) / (2 * width)
                        y_center = (yleft + yright) / (2 * height)
                        w = (xright - xleft) / width
                        h = (yright - yleft) / height
                        new_correct_line = (
                            f"{ID_CLASS_PERSON_NEW} {x_center} {y_center} {w} {h}"
                        )
                        correct_lines.append(
                            new_correct_line
                        )  # пробел из правого слагаемого остается
        with open(txt_path, "w") as file:
            file.write("\n".join(correct_lines))


def prepare_pytorch_KITTI(
    datasets_dir_path,
    ID_CLASSES_PERSON_BEFORE,
    ID_CLASS_PERSON_NEW,
    directory_train_labels,
    directory_train_images,
):
    cwd = os.path.join(datasets_dir_path, "Kitti")

    file_names_train_labels = list_files_in_directory(directory_train_labels)
    file_names_with_person_train = has_person_in_txt(
        ID_CLASSES_PERSON_BEFORE, file_names_train_labels, directory_train_labels
    )
    del_txt_not_in_list(directory_train_labels, file_names_with_person_train)
    del_images_not_in_list_txt(directory_train_images, file_names_with_person_train, '.png')
    relabel_and_del_useless_classes_from_pytorch_KITTI(
        ID_CLASSES_PERSON_BEFORE,
        ID_CLASS_PERSON_NEW,
        file_names_with_person_train,
        directory_train_labels,
        directory_train_images,
    )

    raw_dir = os.path.join(cwd, "raw")
    os.remove(os.path.join(raw_dir, "data_object_image_2.zip"))
    os.remove(os.path.join(raw_dir, "data_object_label_2.zip"))

    shutil.rmtree(os.path.join(raw_dir, "testing"))
    
    training_dir = os.path.join(raw_dir, "training")
    shutil.move(os.path.join(training_dir, "image_2"), os.path.join(training_dir, "images"))
    shutil.move(os.path.join(training_dir, "label_2"), os.path.join(training_dir, "labels"))
    shutil.move(training_dir, os.path.join(cwd, "train"))

    shutil.rmtree(raw_dir)
    
    data = {
        'names': ['person'],
        'nc': 1,
        'train': 'Kitti/train/images',
    }
    with open(os.path.join(cwd, "data.yaml"), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
def transform_kitti(
    datasets_dir_path: str,
    ID_CLASSES_PERSON_BEFORE=ID_CLASSES_PERSON_BEFORE,
    ID_CLASS_PERSON_NEW=ID_CLASS_PERSON_NEW
):
    directory_train_labels = os.path.join(datasets_dir_path, "Kitti", "raw", "training", "label_2")
    directory_train_images = os.path.join(datasets_dir_path, "Kitti", "raw", "training", "image_2")

    print("pytorch KITTI preparing started")
    print("pytorch KITTI preparing started")

    prepare_pytorch_KITTI(
        datasets_dir_path,
        ID_CLASSES_PERSON_BEFORE,
        ID_CLASS_PERSON_NEW,
        directory_train_labels,
        directory_train_images,
    )
    print("pytorch KITTI preparing finished")
