import yaml
from ultralytics import YOLO
import wandb


def prepare_final_yaml():
    dataset_to_train = {}
    # dataset_to_train["Kitti"] = ["train"]
    # dataset_to_train["Pascal-VOC-2012-1"] = ["train"]
    # dataset_to_train["coco-2017"] = ["train"]
    dataset_to_train["open-images-v7"] = ["train"]
    list_train_for_yaml = []

    dataset_to_valid = {}
    # dataset_to_valid["Pascal-VOC-2012-1"] = ["valid"]
    # dataset_to_valid["coco-2017"] = ["valid"]
    dataset_to_valid["open-images-v7"] = ["valid"]
    list_valid_for_yaml = []

    for dataset, subsets in dataset_to_train.items():
        for subset in subsets:
            list_train_for_yaml.append(f"../{dataset}/{subset}/images")
    for dataset, subsets in dataset_to_valid.items():
        for subset in subsets:
            list_valid_for_yaml.append(f"../{dataset}/{subset}/images")

    data_to_yaml = {
        "train": list_train_for_yaml,
        "val": list_valid_for_yaml,
        "nc": 1,
        "names": ["person"],
    }

    with open("data_yolov8.yaml", "w") as file:
        yaml.dump(data_to_yaml, file, default_flow_style=False)


def training():
    prepare_final_yaml()
    # wandb.init(project="yolov8-project", entity="samolet")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="data_yolov8.yaml", epochs=50, imgsz=640, pretrained=True, seed=42
    )


if __name__ == "__main__":
    training()
