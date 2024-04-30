import os

from roboflow import Roboflow

from utils.only_people_from_yolo import load_local_env

def download_prepare_widerperson() -> None:
    load_local_env()

    rf = Roboflow(api_key=os.getenv("API_KEY_ROBOFLOW"))
    project = rf.workspace("zevier-vrnem").project("person-detetion-ma4ln")
    version = project.version(1)
    dataset = version.download("yolov8")

download_prepare_widerperson()