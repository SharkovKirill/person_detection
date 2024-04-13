#!/bin/bash

DATASET_SPLIT=$1

python downloader.py open_images_extended_miap_images_${DATASET_SPLIT}.lst \
       --download_folder=./MIAP/${DATASET_SPLIT}/images/