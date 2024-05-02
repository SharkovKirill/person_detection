#!/bin/bash

GLOBAL_PIDS_ROBOFLOW=""
RESULT=0

function create_env {
    python3 -m venv .venv
    . ./.venv/bin/activate
    pip install -r requirments.pip
}

function disable_env {
    deactivate
}

function make_dir {
    mkdir -p $1
}

function download_dataset_roboflow {
    echo "Starting to download $1"
    curl -s -o /dev/null -L $1 > roboflow.zip
    unzip roboflow.zip > /dev/null 2>&1
    rm roboflow.zip
    echo "End of download of $1"
}

function download_dataset_fiftyone {
    DATASET_NAME=$1
    LABELS=$2
    CLASSES=$3
    fiftyone zoo datasets load ${DATASET_NAME} \
    --kwargs \
        label_types=${LABELS} \
        classes=${CLASSES} \
    --dataset-dir "./${DATASET_NAME}"
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
    "detections Man,Woman,Person"
    "detections person"
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
        GLOBAL_PIDS_ROBOFLOW="$GLOBAL_PIDS_ROBOFLOW $!"
        popd > /dev/null
    done
}

function main {
    create_env

    download_roboflow

    for pid in "${GLOBAL_PIDS_ROBOFLOW}"; do
        wait $pid || let "RESULT=1"
    done

    if [ "$RESULT" == "1" ];
    then
       exit 1
    fi

    download_fiftyone

    python transform_datasets.py

    disable_env
}

main
