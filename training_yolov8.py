import yaml
from ultralytics import YOLO
import wandb


def training():
    # prepare_final_yaml()
    # wandb.init(project="yolov8-project", entity="samolet")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="sampled_datasets/data_yolo8.yaml", epochs=50, imgsz=640, pretrained=False, seed=42, 
    )


if __name__ == "__main__":
    training()
