# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import sys
import threading
import time
from typing import Dict, List
from typing import (Optional, Union)

import matplotlib
import torch
from medutils.medutils import count_parameters
from mlflow import log_metric, log_param, start_run, end_run, log_params, log_artifact
import mlflow
sys.path.append("../..")

import argparse
# import streamlit as st
matplotlib.use('Agg')
from sharednet.modules.set_args import get_args
from sharednet.modules.tool import record_1st, record_2nd, record_cgpu_info
from sharednet.modules.nets import get_net
from sharednet.modules.loss import get_loss
from sharednet.modules.path import Mypath, MypathDataDir
from sharednet.modules.evaluator import get_evaluator
from sharednet.modules.dataset import DataAll
from sharednet.modules.evaluator import get_inferer
import statistics

from monai.metrics import DiceMetric
import monai
import os
import seg_metrics.seg_metrics as sg
import shutil
from medutils.medutils import get_all_ct_names
import pathlib
import glob
from argparse import Namespace

args = get_args()

LogType = Optional[Union[int, float, str]]  # a global type to store immutable variables saved to log files

def get_out_chn(task_name):
    if task_name=="lobe_all":
        out_chn = 6
    elif task_name=="AV_all":
        out_chn = 3
    elif 'all' in task_name and '-' in task_name:  # multi-class dataset' segmentation
        raise Exception(f"The model_names is {task_name} but we have not set the output channel number for multi-class "
                        f"dataset' segmentation. Please reset the model_names.")

    else:
        out_chn = 2
    return out_chn


def mt_netnames(net_names: str) -> List[str]:
    """Get net names from arguments.

    Define the Model, use dash to separate multi net names, do not use ',' to separate it, because ',' can lead to
    unknown error during parse arguments

    Args:
        myargs:

    Returns:
        A list of net names

    """
    #
    net_names = net_names.split('-')
    net_names = [i.lstrip() for i in net_names]  # remove backspace before each net name
    print('net names: ', net_names)

    return net_names


def task_of_model(model_name):
    for task in ['lobe', 'vessel', 'AV', 'liver', 'pancreas']:
        if task in model_name:
            return task


def all_loaders(model_name):
    data = DataAll(model_name, psz=args.psz)
    # if model_name == 'lobe_ll':
    #     data = DataLobeLL()
    # elif model_name == 'lobe_lu':
    #     data = DataLobeLU()
    # elif model_name == 'lobe_ru':
    #     data = DataLobeRU()
    # elif model_name == 'lobe_rm':
    #     data = DataLobeRM()
    # elif model_name == 'lobe_rl':
    #     data = DataLobeRL()
    # elif model_name == 'vessel':
    #     data = DataVessel()
    # elif model_name == 'AV_Artery':
    #     data = DataAVArtery()
    # elif model_name == 'AV_Vein':
    #     data = DataAVVein()
    # elif model_name == 'liver':
    #     data = DataLiver()
    # elif model_name == 'pancreas':
    #     data = DataPancreas()
    # else:
    #     raise Exception(f"Wrong task name {model_name}")

    tr_dl, vd_dl, ts_dl = data.load(cond_flag=args.cond_flag,
                                    same_mask_value=args.same_mask_value,
                                    pps=args.pps,
                                    batch_size=args.batch_size)
    return data, tr_dl, vd_dl, ts_dl

def loop_dl(dl, batch_size):  # convert dict to list, convert wrong batch to right batch
    while True:
        keys = ('image', 'mask', 'cond')
        out_image, out_mask, out_cond = [], [], []

        for ori_batch in dl:  # batch length is batch_size * Croped_patches
            ori_batch_ls = [ori_batch[key] for key in keys]  # [image, mask, cond]
            for image, mask, cond in  zip(*ori_batch_ls):

                out_image.append(image[None])  # a list of image with shape [1, chn,  x, y, z]
                out_mask.append(mask[None])
                out_cond.append(cond[None])
                if len(out_image) >= batch_size:
                    # out_batch_image = torch.Tensor(batch_size, *image.shape[1:])
                    # out_batch_mask = torch.Tensor(batch_size, *mask.shape[1:])
                    # out_batch_cond = torch.Tensor(batch_size, *cond.shape[1:])

                    out_batch_image = torch.cat(out_image, 0)  # [batch_size, chn, x, y, z]
                    out_batch_mask = torch.cat(out_mask, 0)
                    out_batch_cond = torch.cat(out_cond, 0)

                    out_image, out_mask, out_cond = [], [], []  # empty these lists

                    yield out_batch_image, out_batch_mask, out_batch_cond



class Task:
    def __init__(self, model_name, net, out_chn, opt, loss_fun):
        self.model_name = model_name
        task = task_of_model(self.model_name)
        self.data_dir = MypathDataDir(task).data_dir
        self.net = net
        self.mypath = Mypath(args.id, check_id_dir=False, task=task)
        self.out_chn = out_chn
        self.opt = opt
        self.loss_fun = loss_fun
        self.device = torch.device("cuda")
        _, self.tr_dl, self.vd_dl, self.ts_dl = all_loaders(self.model_name)
        self.tr_dl_endless = loop_dl(self.tr_dl, args.batch_size)  # loop training dataset
        self.psz_xy = int(args.psz.split("_")[0])
        self.psz_z = int(args.psz.split("_")[1])

        self.eval_vd = get_evaluator(net, self.vd_dl, self.mypath, self.psz_xy, self.psz_z, args.batch_size, 'valid',
                                        out_chn)
        self.eval_ts = get_evaluator(net, self.ts_dl, self.mypath, self.psz_xy, self.psz_z, args.batch_size, 'test',
                                       out_chn)
        self.accumulate_loss = 0
        self.accumulate_dice_ex_bg = 0

        self.dice_fun_ex_bg = monai.losses.DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.inferer = get_inferer(self.psz_xy, self.psz_z, args.batch_size, 'infer')

    def step(self, step_id, accu_grad, accu_grad_nb):

        self.scaler = torch.cuda.amp.GradScaler()

        # print(f"start a step for {self.model_name}")
        t1 = time.time()
        image, mask, cond = next(self.tr_dl_endless)
        t2 = time.time()
        image, mask, cond = image.to(self.device), mask.to(self.device), cond.to(self.device)
        t3 = time.time()


        if args.amp:
            print('using amp ', end='')
            with torch.cuda.amp.autocast():
                pred = self.net(image,cond)
                loss = self.loss_fun(pred, mask)
                dice_ex_bg = self.dice_fun_ex_bg(pred, mask)
            t4 = time.time()
            if args.grad_accu:
                self.scaler.scale(loss/accu_grad_nb).backward()
                t_bw = time.time()
                if not accu_grad:
                    self.scaler.step(self.opt)
                    t_st = time.time()
                    self.scaler.update()
                    self.opt.zero_grad()
                    print('update grad', end='' )
            else:
                self.scaler.scale(loss).backward()
                t_bw = time.time()
                self.scaler.step(self.opt)
                t_st = time.time()
                self.scaler.update()
                self.opt.zero_grad()
        else:
            print('do not use amp ', end='')
            pred = self.net(image, cond)
            loss = self.loss_fun(pred, mask)
            dice_ex_bg = self.dice_fun_ex_bg(pred, mask)
            t4 = time.time()
            if args.grad_accu:
                (loss / accu_grad_nb).backward()
                if not accu_grad:
                    self.opt.step()
                    self.opt.zero_grad()
                    print('update grad', end='' )

            else:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
        t5 = time.time()

        self.accumulate_loss += loss.item()
        self.accumulate_dice_ex_bg += dice_ex_bg.item()
        if step_id % 200 == 0:
            period = 1 if step_id==0 else 200  # the first accumulate_loss is the first loss
            # todo: change the title if loss function is changed
            log_metric(self.model_name + '_TrDiceInBgTrainBatchIn200Steps', 1 - self.accumulate_loss/period, step_id)
            log_metric(self.model_name + '_TrDiceExBgTrainBatchIn200Steps', 1 - self.accumulate_dice_ex_bg/period, step_id)

            self.accumulate_loss = 0
            self.accumulate_dice_ex_bg = 0
        if step_id <= 1200:  # log the loading time in 1200 steps
            log_metric(self.model_name + 'TimeLoad', t2-t1, step_id)
        if args.amp:
            print(f" {self.model_name} loss: {loss:.3f}, "
                  f"load batch cost: {t2-t1:.1f}, "
                  f"forward costs: {t4-t3:.1f}, "
                  # f"only backward costs: {t_bw-t4:.1f}; "
                  # f"only step costs: {t_st-t_bw:.1f}; "
                  f"backward costs: {t5-t4:.1f}; ", end='' )
        else:
            print(f" {self.model_name} loss: {loss:.3f}, "
                  f"load batch cost: {t2 - t1:.1f}, "
                  f"forward costs: {t4 - t3:.1f}, "
                  f"only update costs: {t5 - t4:.1f}; ", end='')

    # def do_validation(self):
        # if step_id == steps - 1:
        #     print(f"start a test for {self.model_name}")
        #     self.eval_ts.run()

    def infer(self):
        print("start infer")
        # if args.mode == 'train':  # inference the current running project
        model_dir = self.mypath.id_dir  # use the weights from the id_dir
        infer_weights_fpath = best_model_path(model_dir)  # load the saved best model
        self.net.load_state_dict(torch.load(infer_weights_fpath, map_location=self.device))

        keys = ("image","cond")
        self.net.eval()
        if args.infer_data_dir == '':  # prediction may be GLUCOLD or LUNA16 or LOLA11
            prediction_folder = os.path.join(self.mypath.infer_pred_dir)
        else:
            prediction_folder = os.path.join(self.mypath.infer_pred_dir(), args.infer_data_dir.split("/")[-1])
        print(f"prediction_folder: {prediction_folder}")
        saver = monai.data.NiftiSaver(output_dir=prediction_folder,
                                      mode="nearest")  # todo: change mode , mode="nearest"
        # if 1:
        self.infer_loader = self.ts_dl  # test dataset is the infer data
        with torch.no_grad():
            for infer_data in self.infer_loader:
                print(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
                preds = self.inferer((infer_data[keys[0]].to(self.device),
                                      infer_data[keys[1]].to(self.device)), self.net)
                preds = (preds.argmax(dim=1, keepdims=True)).float()
                saver.save_batch(preds, infer_data["image_meta_dict"])

        # copy the saved segmentations into the required folder structure for submission
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)
        # files = glob.glob(os.path.join(prediction_folder,"*", "*.nii.gz"))
        files = get_all_ct_names(os.path.join(prediction_folder, "*"), suffix="_ct_seg")

        for f in files:
            new_name = os.path.basename(f)
            new_name = new_name.split("_ct")[0] + new_name.split("_ct")[1]
            to_name = os.path.join(prediction_folder, new_name)
            shutil.move(f, to_name)
            parent_dir = pathlib.Path(f).parent.absolute()  # remove the empty directory after move its file
            if len(os.listdir(parent_dir)) == 0:
                # removing the file using the os.remove() method
                os.rmdir(parent_dir)

        print(f"predictions moved to {prediction_folder}.")

        all_metrics_ls = sg.write_metrics(labels=[1],  # exclude background
                         gdth_path=self.data_dir,
                         pred_path=prediction_folder,
                         csv_file=prediction_folder + '/metric.csv',
                         metrics=['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd',
                                  'mdsd', 'stdsd'])
        infer_dice = statistics.mean([metric['dice'][0] for metric in all_metrics_ls])
        infer_msd = statistics.mean([metric['msd'][0] for metric in all_metrics_ls])

        log_metric(self.model_name + 'InferDice', infer_dice)
        log_metric(self.model_name + 'InferMSD', infer_msd)


def best_model_path(model_dir):
    model_fpaths = glob.glob(str(model_dir) + "/net_key_metric=*.pt")
    metrics = [float(file_name.split('/')[-1][15:-3]) for file_name in model_fpaths]
    idx = metrics.index(max(metrics))
    return model_fpaths[idx]


def task_dt(model_names, net, out_chn, opt, loss_fun):
    ta_dict: Dict[str, Task] = {}
    for model_name in model_names:
        ta = Task(model_name, net, out_chn, opt, loss_fun)
        ta_dict[model_name] = ta
    return ta_dict


def run(args: Namespace):
    """The main body of the training process.

    Args:
        args: argparse instance

    """
    model_names: List[str] = mt_netnames(args.model_names)

    out_chn = get_out_chn(args.model_names)
    log_param('out_chn', out_chn)
    net = get_net(args.cond_flag, args.cond_method, args.cond_pos, out_chn, args.base)
    net_parameters = count_parameters(net)
    net_parameters = str(round(net_parameters / 1024 / 1024, 2))
    log_param('net_parameters_M', net_parameters)
    device = torch.device("cuda")
    net = net.to(device)
    # if args.infer_ID:  # load trained weights
    #     if len(model_names)==1:
    #         task = model_names[0]
    #     else:
    #         task = 'liver'  # use the weights saved for liver
    #     model_dir = Mypath(args.infer_ID, check_id_dir=False, task= task).id_task_dir
    #     infer_weights_fpath = best_model_path(model_dir)
    #     net.load_state_dict(torch.load(infer_weights_fpath, map_location=device))

    loss_fun = get_loss(loss=args.loss)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # weight decay is L2 weight norm

    ta_dict = task_dt(model_names, net, out_chn, opt, loss_fun)
    ta_ls = list(ta_dict.values())
    vd_dl_all = [data_batch for ta in ta_ls for data_batch in ta.vd_dl]
    mypath_all = Mypath(args.id, check_id_dir=False)
    psz_xy = int(args.psz.split("_")[0])
    psz_z = int(args.psz.split("_")[1])
    eval_vd_all = get_evaluator(net, vd_dl_all, mypath_all, psz_xy, psz_z, args.batch_size, 'valid', out_chn)

    if args.mode != 'infer':
        for step_id in range(args.steps):
            print(f'\nstep number: {step_id}, ', end='' )
            model_idx = 0
            for model_name, ta in ta_dict.items():
                accu_grad = True if model_idx != len(model_names)-1 else False
                ta.step(step_id, accu_grad=accu_grad, accu_grad_nb=len(model_names))
                if step_id % args.valid_period == 0 or step_id == args.steps - 1:
                    print(f"start a valid for {model_name} at time {time.time()}")
                    ta.eval_vd.run()
                    print(f"finish a valid for {model_name} at time {time.time()}")

                    if model_idx==0 and len(model_names)>=1:
                        eval_vd_all.run()
                model_idx += 1

    for model_name, ta in ta_dict.items():
        ta.infer()

    print('Finish all training/validation/testing + metrics!')


def record_artifacts(outfile):
    mythread = threading.currentThread()
    mythread.do_run = True
    if outfile:
        t = 0
        while 1:  # stop signal passed from t
            if mythread.do_run:
                log_artifact(outfile + '_err.txt')
                log_artifact(outfile + '_out.txt')
                if t <= 600:  # 10 minutes
                    period = 10
                    t += period
                else:
                    period = 60
                time.sleep(period)
            else:
                print('record_artifacts do_run is True, let stop the process')
                break

        print('It is time to stop this process: record_artifacts')
        return None
    else:
        print(f"No output file, no log artifacts")
        return None


if __name__ == "__main__":
    log_dict: Dict[str, LogType] = {}  # a global dict to store variables saved to log files

    id, log_dict = record_1st(args)  # write super parameters from set_args.py to record file.


    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        p1 = threading.Thread(target=record_cgpu_info, args=(args.outfile,))
        p1.start()
        p2 = threading.Thread(target=record_artifacts, args=(args.outfile,))
        p2.start()

        log_params(log_dict)
        args.id = id  # do not need to pass id seperately to the latter function
        run(args)

        p1.do_run = False  # stop the thread
        p2.do_run = False  # stop the thread
        p1.join()
        p2.join()
        # time.sleep(80)  # sleep 80 seconds to wait for the finish of other two threads.
        # record_2nd(log_dict=log_dict, args=args)  # write more parameters & metrics to record file.