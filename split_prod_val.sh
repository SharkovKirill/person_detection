#!/bin/bash

DATASET_PATH=$1

CAMERAS_VALID=(
    "camera_6.3_2024"
    "camera_6.6_2024"
    "camera_2_2024"
)

pushd $DATASET_PATH

mkdir -p valid/images/
mkdir -p valid/labels/
mkdir -p train/images/
mkdir -p train/labels/

for (( i=0; i<${#CAMERAS_VALID[@]}; i++ )); do
    images_to_move="images/${CAMERAS_VALID[$i]}*"
    txt_to_move="labels/${CAMERAS_VALID[$i]}*"
    mv $images_to_move ./valid/images/
    mv $txt_to_move ./valid/labels/
done


mv ./images/* ./train/images/
mv ./labels/* ./train/labels/

popd
