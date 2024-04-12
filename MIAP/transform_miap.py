import pandas as pd
import os

for split in ['test', 'train', 'valid']:
    os.makedirs(os.path.join(os.getcwd(), "MIAP", split, "labels"))

for split in ['test', 'train', 'valid']:
    df = pd.read_csv(f"open_images_extended_miap_boxes_{split}.csv")
    df = df[['ImageID', 'XMin', 'XMax', 'YMin', 'YMax']]

    df['XCenter'] = (df['XMin'] + df['XMax']) / 2
    df['YCenter'] = (df['YMin'] + df['YMax']) / 2
    df['Width'] = df['XMax'] - df['XMin']
    df['Height'] = df['YMax'] - df['YMin']

    raw_image = df[['ImageID', 'XCenter', 'YCenter', 'Width', 'Height']].values
    for image in raw_image:
        mode: str = None
        new_line: str = ""
        if os.path.exists(f"./MIAP/{split}/labels/{image[0]}.txt"):
            mode = "a"
            new_line = "\n"
        else:
            mode = "w"
            new_line = ""
        with open(f"./MIAP/{split}/labels/{image[0]}.txt", mode) as f:
            f.write(f"{new_line}0 {image[1]} {image[2]} {image[3]} {image[4]}")
