import torch
import monai
from monai.transforms import LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, \
    RandAffined, RandCropByPosNegLabeld, RandGaussianNoised, RandSpatialCropd, CastToTyped, ToTensord, AsDiscreted, Resize
from torch import nn as nn
from ignite.contrib.handlers import ProgressBar
import os
import numpy as np
from sharednet.modules.custom_inferer import SlidingWindowInfererCond
from typing import Dict, Optional, Union, Tuple, Any, List
from sharednet.modules.tool import MyKeys
from mlflow import log_metric, log_param, start_run, end_run, log_params, log_artifact


from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)


def get_inferer( patch_xy, patch_z, batch_size, mode):
    """returns a sliding window inference instance."""

    patch_size = (patch_xy, patch_xy, patch_z)
    sw_batch_size = batch_size
    overlap = 0.5 if mode=='infer' else 0.1   # todo: less overlap for inferer
    inferer = SlidingWindowInfererCond(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
        device=torch.device('cpu')  # avoid CUDA out of memory
    )
    return inferer

def myprepare_batch(
    batchdata: Dict[str, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    if not isinstance(batchdata, dict):
        raise AssertionError("default prepare_batch expects dictionary input data.")
    return ((batchdata[MyKeys.IMAGE].to(device=device, non_blocking=non_blocking),
             batchdata[MyKeys.COND].to(device=device, non_blocking=non_blocking)),
            batchdata[MyKeys.MASK].to(device=device, non_blocking=non_blocking),
        )


def to_cuda(keys):
    def _wrapper(data):
        if isinstance(data, dict):
            for key in keys:
                data[key] = data[key].to(torch.device("cuda"))
            return data

        elif isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [ [i[k].to(torch.device("cuda")) for i in data] for k in keys]
            return tuple(ret) if len(ret) > 1 else ret[0]
    return _wrapper

def myfrom_engine(keys, first: bool = False, device=torch.device("cpu")):
    """
    Utility function to simplify the `batch_transform` or `output_transform` args of ignite components
    when handling dictionary or list of dictionaries(for example: `engine.state.batch` or `engine.state.output`).
    Users only need to set the expected keys, then it will return a callable function to extract data from
    dictionary and construct a tuple respectively.

    If data is a list of dictionaries after decollating, extract expected keys and construct lists respectively,
    for example, if data is `[{"A": 1, "B": 2}, {"A": 3, "B": 4}]`, from_engine(["A", "B"]): `([1, 3], [2, 4])`.

    It can help avoid a complicated `lambda` function and make the arg of metrics more straight-forward.
    For example, set the first key as the prediction and the second key as label to get the expected data
    from `engine.state.output` for a metric::

        from monai.handlers import MeanDice, from_engine

        metric = MeanDice(
            include_background=False,
            output_transform=from_engine(["pred", "label"])
        )

    Args:
        keys: specified keys to extract data from dictionary or decollated list of dictionaries.
        first: whether only extract sepcified keys from the first item if input data is a list of dictionaries,
            it's used to extract the scalar data which doesn't have batch dim and was replicated into every
            dictionary when decollating, like `loss`, etc.


    """
    def _wrapper(data):
        if isinstance(data, dict):
            return tuple(data[k].to(device) for k in keys)
        elif isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [data[0][k].to(device) if first else [i[k].to(device) for i in data] for k in keys]
            # ret = [x.to(device) for x in ret]
            # for i in ret:
            #     print(f"lengh: {len(i)}")
            #     print(f'shape:{i[0].shape}')
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper


def get_evaluator(net, dl, mypath, patch_xy, patch_z, batch_size, mode, out_chn):
    """Return evaluator. record_val_metrics after one evaluation.

    """
    keys = ("pred", "label")

    val_post_transform = monai.transforms.Compose(
        [ToTensord(keys=keys), AsDiscreted(keys=keys, argmax=(True, False), to_onehot=True, n_classes=out_chn)]
    )
    if mypath.task is None:
        save_dir = mypath.id_dir
        pre = 'AllTasks'
    else:
        save_dir = mypath.id_task_dir
        pre = str(mypath.task)

    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=save_dir,
                        save_dict={"net": net},
                        save_key_metric=True,
                        key_metric_n_saved=3),
    ]


    # output_trans = monai.transforms.Compose([myfrom_engine(keys), to_cuda(keys)])
    if type(dl) is list:
        epoch_length = len(dl)
    else:
        epoch_length = None
    evaluator = monai.engines.SupervisedEvaluator(
        device=torch.device("cuda"),
        val_data_loader=dl,
        epoch_length = epoch_length,
        prepare_batch=myprepare_batch,
        network=net,
        inferer=get_inferer(patch_xy, patch_z, batch_size, mode),
        postprocessing=val_post_transform,
        key_val_metric={"DiceExBg": MeanDice(include_background=False, output_transform =myfrom_engine(keys) )},
        additional_metrics={"DiceInBg": MeanDice(include_background=True, output_transform = myfrom_engine(keys))},
        val_handlers=val_handlers,
        amp=True,
    )


    def record_val_metrics(engine):
        val_log_fpath = str(mypath.metrics_fpath('valid'))
        if os.path.exists(val_log_fpath):
            val_log = np.genfromtxt(val_log_fpath, dtype='str', delimiter=',')
        else:
            val_log = ['DiceExBg', 'DiceInBg']
        dice_ex_bg = round(engine.state.metrics["DiceExBg"], 3)
        dice_inc_bg = round(engine.state.metrics["DiceInBg"], 3)
        val_log = np.vstack([val_log, [dice_ex_bg,dice_inc_bg]])
        np.savetxt(val_log_fpath, val_log, fmt='%s', delimiter=',')
        log_metric(pre + "_" + mode+'DiceExBg', dice_ex_bg, step=len(val_log)-1)
        log_metric(pre + "_" + mode+'DiceInBg', dice_inc_bg, step=len(val_log)-1)

    from ignite.engine import Events
    evaluator.add_event_handler(Events.COMPLETED, record_val_metrics)


    return evaluator
