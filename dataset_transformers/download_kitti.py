import sys

import torchvision

def download_pytorch_KITTI(datasets_dir_path):
    kitti_data = torchvision.datasets.Kitti(root=datasets_dir_path, train=True, download=True)

if __name__ == "__main__":
    download_pytorch_KITTI(sys.argv[1])