import sys
import os

from dataset_transformers.transform_coco import transform_coco
from dataset_transformers.transform_construction_safety import transform_construction_safety
from dataset_transformers.transform_kitti import transform_kitti
from dataset_transformers.transform_openimages7 import transform_openimages7
from dataset_transformers.transform_pascal_voc import transform_pascal_voc
from dataset_transformers.transform_pdf import transform_pdf
from dataset_transformers.transform_widerperson import transform_widerperson

DATASETS_DIR = os.path.join(os.getcwd(), sys.argv[1])

def main() -> int:
    transform_coco(DATASETS_DIR)
    transform_construction_safety(DATASETS_DIR)
    transform_openimages7(DATASETS_DIR)
    transform_pascal_voc(DATASETS_DIR)
    transform_pdf(DATASETS_DIR)
    transform_widerperson(DATASETS_DIR)
    transform_kitti(DATASETS_DIR)

if __name__ == "__main__":
    sys.exit(main())
