#!/bin/bash

DATASET_PATH=$1

CAMERAS_TRAIN=(
    "camera_6.3_2024"
    "camera_6.6_2024"
    "camera_2_2024"
)

pushd $DATASET_PATH

mkdir -p valid/images/
mkdir -p valid/labels/
mkdir -p train/images/
mkdir -p train/labels/

for (( i=0; i<${#CAMERAS_TRAIN[@]}; i++ )); do
    images_to_move="images/${CAMERAS_TRAIN[$i]}*"
    txt_to_move="labels/${CAMERAS_TRAIN[$i]}*"
    mv $images_to_move ./train/images/
    mv $txt_to_move ./train/labels/
done


mv ./images/* ./valid/images/
mv ./labels/* ./valid/labels/

popd
