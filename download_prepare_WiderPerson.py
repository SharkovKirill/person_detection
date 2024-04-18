import os

from roboflow import Roboflow

from utils.only_people_from_yolo import load_local_env

def download_prepare_widerperson() -> None:
    load_local_env()

    rf = Roboflow(api_key=os.getenv("API_KEY_ROBOFLOW"))
    project = rf.workspace("horizon-gvs13").project("widerperson-s1vlr")
    version = project.version(3)
    dataset = version.download("yolov8")

download_prepare_widerperson()