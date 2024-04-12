import os
from dotenv import load_dotenv


def load_local_env():
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)


def list_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


def has_person_in_txt(ID_CLASSES_PERSON_BEFORE: tuple, file_names, dir_path: str):
    file_names_with_person = []
    for file_name in file_names:
        file_path = dir_path + f"/{file_name}"
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith(ID_CLASSES_PERSON_BEFORE):
                    file_names_with_person.append(file_name)
                    break
    return file_names_with_person



def del_txt_not_in_list(directory, file_names_txt):
    # Удаляет txt с аннотациями в которых не встречаются люди
    for file_name in os.listdir(directory):
        if file_name not in file_names_txt:
            file_path = directory + f"/{file_name}"
            os.remove(file_path)


def del_images_not_in_list_txt(directory, file_names_txt, pictures_type:str=".jpg"):
    # Удаляет картинки в которых не встречаются люди
    for file_name in os.listdir(directory):
        file_name_to_txt = file_name.rstrip(pictures_type) + ".txt"
        if file_name_to_txt not in file_names_txt:
            file_path = directory + f"/{file_name}"
            os.remove(file_path)



def relabel_and_del_useless_classes_from_yolo8(
    ID_CLASSES_PERSON_BEFORE: tuple, ID_CLASS_PERSON_NEW, file_names, dir_path: str
):
    for file_name in file_names:
        file_path = dir_path + f"/{file_name}"
        correct_lines = []
        with open(file_path, "r") as file:
            all_lines = file.read().split("\n")
            for line in all_lines:
                for particular_word in ID_CLASSES_PERSON_BEFORE:
                    if line.startswith(particular_word):
                        correct_lines.append(
                        str(ID_CLASS_PERSON_NEW)
                        + line[len(particular_word) :]
                    ) # пробел из правого слагаемого остается
        with open(file_path, "w") as file:
            file.write("\n".join(correct_lines))
