# Person Fisheye Detection

import os

from roboflow import Roboflow

from utils.only_people_from_yolo import load_local_env

def download_prepare_pdf() -> None:
    load_local_env()

    rf = Roboflow(api_key="dWeTAd2AQjsWYnrlClGq")
    project = rf.workspace("ltp-tvqcg").project("person-detection-fisheye")
    version = project.version(1)
    dataset = version.download("yolov8")

download_prepare_pdf()