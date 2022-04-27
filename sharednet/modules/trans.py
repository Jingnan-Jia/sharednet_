import monai
import copy
from typing import Dict, Optional, Union, Hashable

import monai
import numpy as np
import torch
from monai.data import NibabelReader, ITKReader
from monai.transforms import LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, \
    RandAffined, RandCropByPosNegLabeld, RandGaussianNoised, CastToTyped, ToTensord, RandSpatialCropSamplesd
from monai.transforms import Transform
from mlflow import log_metric, log_param

TransInOut = Dict[Hashable, Optional[Union[np.ndarray, torch.Tensor, str, int]]]


def tolist(model_names):
    ls = model_names.split('-')
    return ls


class FilterMask(Transform):

    def __init__(self, key, model_name, cond_flag, same_mask_value):
        self.model_name = model_name
        self.key = key
        self.cond_flag = cond_flag

        self.same_mask_value = same_mask_value
        self.mask_value_original = {'lobe_ru': 1,
                                    'lobe_rm': 2,
                                    'lobe_rl': 3,
                                    'lobe_lu': 4,
                                    'lobe_ll': 5,
                                    'lobe_all': [1, 2, 3, 4, 5],
                                    'lung': 'positive',

                                    'AV_artery': 1,
                                    'AV_vein': 2,
                                    'AV_all': [1, 2],
                                    'vessel': 'positive',  # label='MergePositiveLabels',

                                    'liver': 1,
                                    'pancreas': 1}

        self.mask_value_target = {'lobe_ru': 1,
                                  'lobe_rm': 2,
                                  'lobe_rl': 3,
                                  'lobe_lu': 4,
                                  'lobe_ll': 5,
                                  'lobe_all': 6,
                                  'lung': 7,

                                  'AV_artery': 8,
                                  'AV_vein': 9,
                                  'AV_all': 10,
                                  'vessel': 11,  # label='MergePositiveLabels',

                                  'liver': 12,
                                  'pancreas': 13}

        log_param('mask_value_target', str(self.mask_value_target))
        log_param('mask_value_original', str(self.mask_value_original))

    def __call__(self, data: TransInOut) -> TransInOut:
        data_tmp = copy.deepcopy(data[self.key])
        ori_label = self.mask_value_original[self.model_name]
        if type(ori_label) is int:
            data_tmp[np.where(data_tmp != ori_label)] = 0  # select the background
        elif ori_label == 'positive':
            data_tmp[np.where(data_tmp <= 0)] = 0  # select the background
        elif type(ori_label) is list:
            pass
        else:
            raise Exception(f"wrong original label: {ori_label}")

        if self.same_mask_value:
            data_tmp[np.where(data_tmp != 0)] = 1  # set foreground
        else:
            data_tmp[np.where(data_tmp != 0)] = self.mask_value_target[self.model_name]  # set foreground

        data[self.key] = data_tmp

        if self.cond_flag:
            data['cond'] = self.mask_value_target[self.model_name]  # add another key
        else:
            data['cond'] = 0  # set cond 0 if no condition. it is convenient to batch data in pytorch.dataloader.

        return data


def get_xforms(model_name, cond_flag, same_mask_value, patch_xy, patch_z, tsp_xy, tsp_z, pps, mode: str = "train",
               keys=("image", "mask")):
    """returns a composed transform for train/val/infer."""
    nibabel_reader = NibabelReader()
    itk_reader = ITKReader()
    xforms = [
        LoadImaged(keys),  # .nii, .nii.gz [.mhd, .mha LoadImage]
        FilterMask(keys[1], model_name, cond_flag, same_mask_value),  # new feature in sharednet
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(tsp_xy, tsp_xy, tsp_z), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                # SpatialPadd(keys, spatial_size=(args.patch_xy, args.patch_xy, args.patch_z), mode="minimum"),
                # ensure at least HTxHT*z
                RandAffined(
                    keys,
                    prob=0.7,
                    rotate_range=(0.05, 0.05),  # when changing it, do not forget to update the log_params
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                SpatialPadd(keys, spatial_size=(patch_xy, patch_xy, patch_z), mode="minimum"),
                RandSpatialCropSamplesd(keys, random_size=False, roi_size=(patch_xy, patch_xy, patch_z), num_samples=pps),
                # RandCropByPosNegLabeld(keys, label_key=keys[1],
                #                        spatial_size=(patch_xy, patch_xy, patch_z),
                #                        num_samples=pps),
                SpatialPadd(keys, spatial_size=(patch_xy, patch_xy, patch_z), mode="minimum"),

                # todo: num_samples
                RandGaussianNoised(keys[0], prob=0.3, std=0.01),

                # RandFlipd(keys, spatial_axis=0, prob=0.5),
                # RandFlipd(keys, spatial_axis=1, prob=0.5),
                # RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    elif mode in ["valid", "test"]:
        dtype = (np.float32, np.uint8)
    elif mode == "infer":
        dtype = (np.float32,)
    else:
        raise Exception(f"mode {mode} is not correct, please set mode as 'train', 'val' or 'infer'. ")
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)