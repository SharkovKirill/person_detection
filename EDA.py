from utils.vizualize_bboxes import show_bb_yolo, save_hists

show_bb_yolo(
    "KITTI",
    "./Kitti/train/",
    cols=5,
    rows=2,
    shuffle=True,
    data_type="train",
    figsize=(40, 20),
    pictures_type=".png"
)

show_bb_yolo(
    "VOC",
    "./Pascal-VOC-2012-1/train/",
    cols=5,
    rows=2,
    shuffle=True,
    data_type="train",
    figsize=(40, 20),
)

show_bb_yolo(
    "VOC",
    "./Pascal-VOC-2012-1/valid/",
    cols=5,
    rows=2,
    shuffle=True,
    data_type="valid",
    figsize=(40, 20),
)

save_hists("KITTI", "./Kitti/train/", "train", figsize=(15, 6))
save_hists("VOC", "./Pascal-VOC-2012-1/train/", "train", figsize=(15, 6))
save_hists("VOC", "./Pascal-VOC-2012-1/valid/", "train", figsize=(15, 6))