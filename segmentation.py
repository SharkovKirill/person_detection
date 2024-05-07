from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
from numpy import ndarray


def initialize_sam(sam_weights_path: str, sam_model_type: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_weights_path).to(
        device=device
    )

    return sam


def bboxes_SAM_from_yolo(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return np.array([x1, y1, x2, y2])


def boxes_sam_one_pic_io(one_image_path: str, one_label_path: str):
    image = cv2.imread(one_image_path)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape  # height - высота, width - ширина
    elif len(image.shape) == 2:
        height, width = image.shape
    with open(one_label_path, "r") as bb_file:
        boxes = bb_file.read().split("\n")
    default_boxes = []  # list of lists
    for one_box in boxes:
        box_list = list(map(float, one_box.split(" ")))
        default_boxes.append(
            bboxes_SAM_from_yolo(
                box_list[1], box_list[2], box_list[3], box_list[4], width, height
            )
        )  # changing the order for augmentation
    return default_boxes


def boxes_sam_one_pic(
    image,  # np.array H*W*C
    correct_lines_float: List[
        List[float]
    ],  # correct order like List[List[x_center, y_center, w, h]]
):
    height, width = image.shape[0], image.shape[1]

    default_boxes = []  # list of lists
    for box_list in correct_lines_float:
        default_boxes.append(
            bboxes_SAM_from_yolo(
                box_list[1], box_list[2], box_list[3], box_list[4], width, height
            )
        )  # changing the order for augmentation
    return default_boxes


def xyxy_to_yolov8(x1, y1, x2, y2, height, width):
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    x_center = (x2 + x1) / (2 * width)
    y_center = (y2 + y1) / (2 * height)
    return x_center, y_center, w, h


def segment_one_image(
    sam: Sam,
    image_rgb: ndarray,
    correct_lines_float,
    figsize: tuple = None,
    show_plots: bool = False,
):
    ID_CLASS_PERSON_NEW = int(correct_lines_float[0][0])
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image_rgb)
    bboxes_one_image = boxes_sam_one_pic(image_rgb, correct_lines_float)
    if show_plots:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        masks_all = np.zeros((image_rgb.shape[0], image_rgb.shape[1]))

    correct_new_boxes = []

    for one_bbox_aug in bboxes_one_image:
        masks, _, _ = mask_predictor.predict(box=one_bbox_aug, multimask_output=True)
        if not (masks[0] == 0).all(): # если объект нашелся
            if show_plots:
                box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
                mask_annotator = sv.MaskAnnotator(
                    color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX
                )

            detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
            detections = detections[detections.area == np.max(detections.area)]
            x_center, y_center, w, h = xyxy_to_yolov8(
                detections.xyxy[0][0],
                detections.xyxy[0][1],
                detections.xyxy[0][2],
                detections.xyxy[0][3],
                image_rgb.shape[0],
                image_rgb.shape[1],
            )
            new_correct_line = [int(ID_CLASS_PERSON_NEW), x_center, y_center, w, h]
            correct_new_boxes.append(new_correct_line)
            if show_plots:
                masks_all += detections.mask.squeeze()

                source_image = box_annotator.annotate(
                    scene=image_bgr.copy(), detections=detections, skip_label=True
                )
                segmented_image = mask_annotator.annotate(
                    scene=image_bgr.copy(), detections=detections
                )

                sv.plot_images_grid(
                    images=[source_image, segmented_image],
                    grid_size=(1, 2),
                    titles=["source image", "segmented image"],
                    size=figsize,
                )
                
    if show_plots:
        image_to_show = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        red_color = np.array([255, 0, 0], dtype=np.uint8)
        image_to_show[masks_all == 1] = red_color
        if figsize is not None:
            plt.figure(figsize=(20, 10))
        plt.imshow(image_to_show)
    return correct_new_boxes, image_rgb
