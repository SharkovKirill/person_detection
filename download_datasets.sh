#!/bin/bash

GLOBAL_PIDS_TO_WAIT=""
RESULT=0

function create_env {
    python3 -m venv .venv
    . ./.venv/bin/activate
    pip install -r requirments.pip
}

function disable_env {
    source deactivate
}

function make_dir {
    mkdir -p $1
}

function download_dataset_roboflow {
    echo "Starting to download $1"
    curl -s -L $1 > roboflow.zip
    unzip roboflow.zip > /dev/null 2>&1
    rm roboflow.zip
    echo "End of download of $1"
}

function download_dataset_fiftyone {
    DATASET_NAME=$1
    LABELS=$2
    CLASSES=$3
    MAX_SAMPLES=($(echo ${4//,/ }))
    SPLITS=($(echo ${5//,/ }))

    for (( j=0; j<${#SPLITS[@]}; j++ )); do
        fiftyone zoo datasets load ${DATASET_NAME} \
        --split ${SPLITS[j]} \
        --kwargs \
            label_types=${LABELS} \
            classes=${CLASSES} \
            max_samples=${MAX_SAMPLES[j]} \
        --dataset-dir "./${DATASET_NAME}"
    done
}

roboflow_dataset_names=(
    Pascal-VOC-2012-1
    Construction-Site-Safety-30
    WiderPerson
    Person-Detection-Fisheye-1
)
roboflow_dataset_links=(
    "https://universe.roboflow.com/ds/Xz5sNyOaGn?key=N9c9Cn72b3"
    "https://universe.roboflow.com/ds/H8CiJgKfTH?key=JCKp8eGta3"
    "https://universe.roboflow.com/ds/M6pHO7NJNn?key=sLKOSx1d65"
    "https://universe.roboflow.com/ds/fSi0GUEvlT?key=r2imbWQJM7"
)

fiftyone_dataset_names=(
    open-images-v7
    coco-2017
)
fiftyone_dataset_args=(
    "detections Man,Woman,Person 2000,3000,10000 validation,test,train"
    "detections person 2000,13000 validation,train"
)

sam_weights=(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)

function download_fiftyone {
    for (( i=0; i<${#fiftyone_dataset_names[@]}; i++ )); do
        download_dataset_fiftyone ${fiftyone_dataset_names[$i]} ${fiftyone_dataset_args[$i]}
    done
}

function download_roboflow {
    for (( i=0; i<${#roboflow_dataset_names[@]}; i++ )); do
        make_dir ${roboflow_dataset_names[i]}
        pushd ${roboflow_dataset_names[i]} > /dev/null

        download_dataset_roboflow ${roboflow_dataset_links[$i]} &
        GLOBAL_PIDS_TO_WAIT="$GLOBAL_PIDS_TO_WAIT $!"
        popd > /dev/null
    done
}

function download_sam_weights_async {
    echo "Starting to download $1"
    curl -s -O $1
    echo "End of download of $1"
}

function download_sam_weights {
    for (( i=0; i<${#sam_weights[@]}; i++ )); do
        download_sam_weights_async ${sam_weights[i]} &
    done
}

function main {
    DATASETS_DIR="datasets"

    create_env

    make_dir sam_weights
    pushd sam_weights > /dev/null

    download_sam_weights

    popd > /dev/null

    make_dir ${DATASETS_DIR}
    pushd ${DATASETS_DIR} > /dev/null

    download_roboflow

    for pid in "${GLOBAL_PIDS_TO_WAIT}"; do
        wait $pid || let "RESULT=1"
    done

    if [ "$RESULT" == "1" ];
    then
       exit 1
    fi

    download_fiftyone

    popd > /dev/null

    python dataset_transformers/download_kitti.py ${DATASETS_DIR}
    python transform_datasets.py ${DATASETS_DIR}

}

main
