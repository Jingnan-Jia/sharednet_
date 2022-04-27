from medutils.medutils import get_all_ct_names, load_itk, save_itk
from pathlib import Path
from glob import glob
import csv
import pandas as pd
import numpy as np


if __name__ == '__main__':
    data_folder = Path("/data/jjia/mt/data/pancreas")
    image_list = sorted(glob(str(data_folder.joinpath("*_ct.nii.gz"))))
    mask_list = sorted(glob(str(data_folder.joinpath("*_seg.nii.gz"))))
    print(f"image list:{image_list}")
    print(f"mask list:{mask_list}")

    df = pd.DataFrame(columns=["file_basename", "size_z", "size_y", "size_x", "spacing_z", "spacing_y", "spacing_x", "FOV_z", "min", "max"])
    for name_ct, name_gt in zip(image_list, mask_list):
        image, origin, spacing_image = load_itk(name_ct, require_ori_sp=True)  # origin/spacing order z,y,x
        size = image.shape
        mask, origin, spacing_mask = load_itk(name_gt, require_ori_sp=True)
        size_gt = mask.shape
        if size != size_gt:
            print(f"image name: {name_ct}, image shape: {size}, mask shape: {size_gt}")
        if (spacing_image != spacing_mask).any():
            print(f"image name: {name_ct}, image spacing: {spacing_image}, mask spacing: {spacing_mask}")
            save_itk(name_gt.replace('.nii.gz', 'newspace.nii.gz'), mask, origin, spacing_image)
            print(f"successfully save mask with new spacing {name_gt.replace('.nii.gz', 'newspace.nii.gz')}")

        new_row = {"file_basename": str(Path(name_ct).name),
                   "size_z": size[0],
                   "size_y": size[1],
                   "size_x": size[2],

                   "spacing_z": spacing_image[0],
                   "spacing_y": spacing_image[1],
                   "spacing_x": spacing_image[2],
                   "FOV_z": size[0] * spacing_image[0],
                   "min": np.min(image),
                   "max": np.max(image)}
        df = df.append(new_row, ignore_index=True)

    # df.to_csv("/data/jjia/sharednet/sharednet/results/dataset_statistics_liver.csv")

