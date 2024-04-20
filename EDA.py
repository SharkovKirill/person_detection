from utils.vizualize_bboxes import show_bb_yolo, save_hists

show_bb_yolo(
    "KITTI",
    "./Kitti/train/",
    cols=5,
    rows=2,
    shuffle=True,
    data_type="train",
    figsize=(40, 20),
    pictures_type=".png",
)
save_hists("KITTI", "./Kitti/train/", "train", figsize=(15, 6))


for subset in ["train", "valid"]:
    show_bb_yolo(
        "VOC",
        f"./Pascal-VOC-2012-1/{subset}/",
        cols=5,
        rows=2,
        shuffle=True,
        data_type=subset,
        figsize=(40, 20),
    )
    save_hists("VOC", f"./Pascal-VOC-2012-1/{subset}/", subset, figsize=(15, 6))


for subset in ["train", "valid"]:
    show_bb_yolo(
        "COCO",
        f"./coco-2017/{subset}/",
        cols=5,
        rows=2,
        shuffle=True,
        data_type=subset,
        figsize=(40, 20),
    )
    save_hists("COCO", f"./coco-2017/{subset}/", subset, figsize=(15, 6))


# for subset in ["train", "valid", "test"]:
#     show_bb_yolo(
#         "WiderPerson",
#         f"./WiderPerson-3/{subset}/",
#         cols=5,
#         rows=2,
#         shuffle=True,
#         data_type=subset,
#         figsize=(40, 20),
#     )
#     save_hists("WiderPerson", f"./WiderPerson-3/{subset}/", subset, figsize=(15, 6))


for subset in ["train", "valid", "test"]:
    show_bb_yolo(
        "open-images-v7",
        f"./open-images-v7/{subset}/",
        cols=5,
        rows=2,
        shuffle=True,
        data_type=subset,
        figsize=(40, 20),
    )
    save_hists(
        "open-images-v7", f"./open-images-v7/{subset}/", subset, figsize=(15, 6)
    )
