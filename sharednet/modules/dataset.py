from typing import Union, Sequence, Optional
from sharednet.modules.path import Mypath, MypathDataDir
import csv
import glob
import logging
import os
import random
import shutil
import time
from shutil import copy2
from typing import List, Optional, Union, Dict, Tuple
import seg_metrics.seg_metrics as sg
import pathlib
import monai
import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from monai.data import NibabelReader, ITKReader, Dataset
from monai.handlers import CheckpointSaver, MeanDice, ValidationHandler, StatsHandler
from monai.transforms import LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, \
    RandAffined, RandCropByPosNegLabeld, RandGaussianNoised, RandSpatialCropd, CastToTyped, ToTensord, AsDiscreted, \
    Resize
from medutils.medutils import get_all_ct_names, load_itk, save_itk
from typing import Callable, List, Optional, Sequence, Union
from monai.transforms import RandGaussianNoise, Transform, RandomizableTransform, ThreadUnsafe
from typing import Dict, Optional, Union, Hashable, Sequence
import copy
from mlflow import log_metric, log_param, log_params, log_artifacts
from sklearn.model_selection import KFold
from monai.data.utils import pad_list_data_collate

TransInOut = Dict[Hashable, Optional[Union[np.ndarray, torch.Tensor, str, int]]]
from sharednet.modules.trans import get_xforms


def get_file_names(data_dir, return_mode=('train', 'valid')):
    """Return 2 lists of training and validation file names."""
    keys = ("image", "mask")
    print(f'data dir: {data_dir}')

    ct_names = np.array(get_all_ct_names(data_dir, suffix="_ct"))
    gt_names = np.array(get_all_ct_names(data_dir, suffix="_seg"))
    assert len(ct_names) == len(gt_names)

    # if self.main_net_name!='lobe': # pat_28 should be in testing dataset.
    SEED = 47
    log_param('data_shuffle_seed', SEED)
    # random.seed(SEED)
    # random.shuffle(ct_names)
    # random.seed(SEED)
    # random.shuffle(gdth_names)
    total_folds = 4
    current_fold = 1
    kf = KFold(n_splits=total_folds, shuffle=True, random_state=SEED)  # for future reproduction
    kf_list = list(kf.split(ct_names))
    tr_vd_idx, ts_idx = kf_list[current_fold - 1]

    ct_tr_vd, ct_ts = ct_names[tr_vd_idx], ct_names[ts_idx]
    gt_tr_vd, gt_ts = gt_names[tr_vd_idx], gt_names[ts_idx]
    train_frac, val_frac = 0.8, 0.2

    data_len = len(ct_tr_vd)
    tr_nb = round(train_frac * data_len)
    if tr_nb == data_len:
        tr_nb -= 1
    if tr_nb == 0:
        raise Exception(f"training number is 0 !")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(ct_tr_vd[:tr_nb], gt_tr_vd[:tr_nb])]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(ct_tr_vd[tr_nb:], gt_tr_vd[tr_nb:])]
    ts_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(ct_ts, gt_ts)]
    print(f"train_files: {train_files}")
    print(f"valid_files: {val_files}")
    print(f"test_files: {ts_files}")

    if return_mode == ('train', 'valid'):
        return train_files, val_files
    elif return_mode == ('train', 'valid', 'test'):
        return train_files, val_files, ts_files
    else:
        raise Exception(f"please give correct mode ('train', 'valid') or ('train', 'valid', 'test'), "
                        f"instead of {return_mode}")


def mydataloader(model_name, cond_flag, same_mask_value, patch_xy, patch_z, tsp_xy, tsp_z, pps, data_dir,
                 return_mode, load_workers, cache,
                 batch_size):
    """Return train (and valid) dataloader.


    Args:
        model_name: a single name
        return_mode: ('train'), or ('train', 'valid') or ('train', 'valid', 'test')

    """
    ct_name_list: List[str]
    gdth_name_list: List[str]
    train_files, valid_files, test_files = get_file_names(data_dir=data_dir, return_mode=return_mode)
    # train_files, valid_files, test_files = train_files[:2], valid_files[:1], test_files[:1]
    # train_files, val_files = train_files[:2], val_files[:2]
    loaders = []
    if 'train' in return_mode:
        train_transforms = get_xforms(model_name, cond_flag, same_mask_value, patch_xy, patch_z, tsp_xy, tsp_z, pps, mode="train")
        if cache:
            train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms,
                                               num_workers=load_workers, cache_rate=1)
        else:
            train_ds = Dataset(data=train_files, transform=train_transforms, )

        train_loader = monai.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=load_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            # collate_fn=pad_list_data_collate,
        )
        loaders.append(train_loader)

    if 'valid' in return_mode:
        # create a validation data loader
        val_transforms = get_xforms(model_name, cond_flag, same_mask_value, patch_xy, patch_z, tsp_xy, tsp_z, pps, mode="valid")
        print('valid files:', valid_files)
        val_ds = monai.data.CacheDataset(data=valid_files, transform=val_transforms, num_workers=load_workers)
        val_loader = monai.data.DataLoader(
            val_ds,
            batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
            num_workers=load_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            # collate_fn=pad_list_data_collate

        )
        loaders.append(val_loader)


    if 'test' in return_mode:
        test_transforms = get_xforms(model_name, cond_flag, same_mask_value, patch_xy, patch_z, tsp_xy, tsp_z, pps, mode="test")
        print(f'test files: {test_files}')
        test_ds = monai.data.CacheDataset(data=test_files, transform=test_transforms, num_workers=load_workers)
        test_loader = monai.data.DataLoader(
            test_ds,
            batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
            num_workers=load_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        # collate_fn=pad_list_data_collate
        )
        loaders.append(test_loader)


    if return_mode in (('train'), ('train', 'valid'), ('train', 'valid', 'test')):
        return loaders
    else:
        raise Exception(f"return mode shoud be one of the following 3 choices: "
                        f"('train'), ('train', 'valid'), ('train', 'valid', 'test') "
                        f"However the current return mode is: {return_mode}")


class Data:
    """
    Data class for different tasks
    """

    def __init__(self, task: str, tsp: Optional[str], psz: str, workers: int):
        self.task = task
        if tsp is not None:
            self.tsp_xy = float(tsp.split("_")[0])
            self.tsp_z = float(tsp.split("_")[1])
        else:
            self.tsp_xy, self.tsp_z = None, None

        self.psz_xy = int(psz.split("_")[0])
        self.psz_z = int(psz.split("_")[1])
        self.data_dir = MypathDataDir(task).data_dir
        self.workers = workers

        log_params({f"{task}_tsp_xy": self.tsp_xy,
                    f"{task}_tsp_z": self.tsp_z,
                    f"{task}_psz_xy": self.psz_xy,
                    f"{task}_psz_z": self.psz_z})

    def load(self, cond_flag, same_mask_value, pps, batch_size, return_mode=('train', 'valid', 'test'), cache=True):
        all_loaders = mydataloader(self.task, cond_flag, same_mask_value, self.psz_xy, self.psz_z, self.tsp_xy, self.tsp_z,
                                   pps, self.data_dir, return_mode, self.workers, cache, batch_size)

        return all_loaders


class DataAll(Data):
    """
    Data class for all tasks
    """

    def __init__(self,
                 task: str,
                 psz: str = "128_128"
                 ):
        workers = 12
        log_param('workers', workers)
        if 'lobe' in task:
            tsp: Optional[str] = "1.5_2.5"
        elif 'vessel' in task or 'AV' in task:
            tsp = None
        elif 'liver' in task or 'pancreas' in task:
            tsp = "1.5_1"
        else:
            raise Exception(f"wrong task name: {task}")
        super().__init__(task, tsp, psz, workers)
#
# class DataLobe(Data):
#     """
#     Data class for lobe task
#     """
#
#     def __init__(self,
#                  task: str,
#                  tsp: str = "1.5_2.5",
#                  psz: str = "144_96"):
#         super().__init__(task, tsp, psz)
#
#
# class DataLobeLU(DataLobe):
#     """
#     Data class for left-upper lobe task
#     """
#
#     def __init__(self, task: str = "lobe_lu"):
#         super().__init__(task)
#
#
# class DataLobeLL(DataLobe):
#     """
#     Data class for left-lower lobe task
#     """
#
#     def __init__(self, task: str = "lobe_ll"):
#         super().__init__(task)
#
#
# class DataLobeRU(DataLobe):
#     """
#     Data class for right-upper lobe task
#     """
#
#     def __init__(self, task: str = "lobe_ru"):
#         super().__init__(task)
#
#
# class DataLobeRM(DataLobe):
#     """
#     Data class for right-middle lobe task
#     """
#
#     def __init__(self, task: str = "lobe_rm"):
#         super().__init__(task)
#
#
# class DataLobeRL(DataLobe):
#     """
#     Data class for right-lower lobe task
#     """
#
#     def __init__(self, task: str = "lobe_rl"):
#         super().__init__(task)
#
#
# class DataLobeALL(DataLobe):
#     """
#     Data class for all lobes task
#     """
#
#     def __init__(self, task: str = "lobe_all"):
#         super().__init__(task)
#
#
# class DataVessel(Data):
#     """
#     Data class for vessel task
#     """
#
#     def __init__(self,
#                  task: str = "vessel",
#                  tsp=None,
#                  psz="144_96"):
#         super().__init__(task, tsp, psz)
#
#
# class DataAVArtery(Data):
#     """
#     Data class for vessel task
#     """
#
#     def __init__(self,
#                  task: str = "AV_Artery",
#                  tsp=None,
#                  psz="144_96"):
#         super().__init__(task, tsp, psz)
#
#
# class DataAVVein(Data):
#     """
#     Data class for vessel task
#     """
#
#     def __init__(self,
#                  task: str = "AV_Vein",
#                  tsp=None,
#                  psz="144_96"):
#         super().__init__(task, tsp, psz)
#
#
# class DataLiver(Data):
#     """
#     Data class for vessel task
#     """
#
#     def __init__(self,
#                  task: str = "liver",
#                  tsp="1.5_1",
#                  psz="144_96"):
#         super().__init__(task, tsp, psz)
#
#
# class DataPancreas(Data):
#     """
#     Data class for vessel task
#     """
#
#     def __init__(self,
#                  task: str = "pancreas",
#                  tsp="1.5_1",
#                  psz="144_96"):
#         super().__init__(task, tsp, psz)


