import sys

from dataset_transformers.transform_coco import transform_coco
from dataset_transformers.transform_construction_safety import transform_construction_safety
from dataset_transformers.transform_kitti import transform_kitti
from dataset_transformers.transform_openimages7 import transform_openimages7
from dataset_transformers.transform_pascal_voc import transform_pascal_voc
from dataset_transformers.transform_pdf import transform_pdf
from dataset_transformers.transform_widerperson import transform_widerperson

def main() -> int:
    transform_coco()
    transform_construction_safety()
    transform_openimages7()
    transform_pascal_voc()
    transform_pdf()
    transform_widerperson()
    transform_kitti()

if __name__ == "__main__":
    sys.exit(main())
