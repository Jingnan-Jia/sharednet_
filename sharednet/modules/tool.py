import sys
sys.path.append("../..")

import argparse
import datetime
import os
import shutil
import time
from typing import Union, Tuple
import threading

import numpy as np
import nvidia_smi
import pandas as pd
from filelock import FileLock
from sharednet.modules.path import Mypath, MypathBase
from pathlib import Path
from mlflow import log_metric, log_param, log_artifacts, log_params
import psutil

class MyKeys:
    """The keys I used in this project."""
    IMAGE = "image"
    MASK = "mask"
    COND = "cond"
    PRED = "pred"
    LOSS = "loss"


def get_loss_min(fpath: str) -> float:
    """Get minimum loss from fpath.

    Args:
        fpath: A csv file in which the loss at each epoch is recorded

    Returns:
        Minimum loss value

    Examples:
        :func:`mt.mymodules.tool.get_loss_min('1635031365_299train.csv')`

    """
    loss = pd.read_csv(fpath)
    mae = min(loss['ave_tr_loss'].to_list())
    return mae



def eval_net_mae(mypath: Path, mypath2: Path) -> float:
    """Copy trained model and loss log to new directory and get its valid_mae_best.

    Args:
        mypath: Current experiment Path instance
        mypath2: Trained experiment Path instance, if mypath is empty, copy files from mypath2 to mypath

    Returns:
        valid_mae_minimum

    Examples:
        :func:`ssc_scoring.run.train` and :func:`ssc_scoring.run_pos.train`

    """
    shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
    for mo in ['train', 'valid', 'test']:
        try:
            shutil.copy(mypath2.metrics_fpath(mo), mypath.metrics_fpath(mo))  # make sure there is at least one model
        except FileNotFoundError:
            print(f'Cannot find the metrics of this mode: {mo}, pass it')
            pass
    valid_mae_best = get_loss_min(mypath2.metrics_fpath('valid'))
    print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
    return valid_mae_best


def add_best_metrics(df: pd.DataFrame,
                     mypath,
                     mypath2,
                     index: int) -> pd.DataFrame:
    """Add best metrics: loss, mae (and mae_end5 if possible) to `df` in-place.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)
        mypath: Current Path instance
        mypath2: Old Path instance, if the loss file can not be find in `mypath`, copy it from `mypath2`
        index: Which row the metrics should be writen in `df`

    Returns:
        `df`

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    modes = ['train', 'valid', 'test']
    metrics_min = 'loss'
    df.at[index, 'metrics_min'] = metrics_min

    for mode in modes:
        lock2 = FileLock(mypath.loss(mode) + ".lock")
        # when evaluating/inference old models, those files would be copied to new the folder
        with lock2:
            try:
                loss_df = pd.read_csv(mypath.loss(mode))
            except FileNotFoundError:  # copy loss files from old directory to here

                shutil.copy(mypath2.loss(mode), mypath.loss(mode))
                try:
                    loss_df = pd.read_csv(mypath.loss(mode))
                except FileNotFoundError:  # still cannot find the loss file in old directory, pass this mode
                    continue

            best_index = loss_df[metrics_min].idxmin()
            loss = loss_df['loss'][best_index]

        df.at[index, mode + '_loss'] = round(loss, 2)
    return df


def write_and_backup(df: pd.DataFrame, record_file: str, mypath) -> None:
    """Write `df` to `record_file` and backup it to `mypath`.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)
        record_file: A file in hard disk saving df
        mypath: Path instance

    Returns:
        None. Results are saved to disk.

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st` and :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    df.to_csv(record_file, index=False)
    shutil.copy(record_file, os.path.join(mypath.result_dir, 'cp_' + os.path.basename(record_file)))
    df_lastrow = df.iloc[[-1]]
    df_lastrow.to_csv(mypath.id_dir.joinpath(record_file.name), index=False)  # save the record of the current ex


def fill_running(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the old record of completed experiments if the state of them are still 'running'.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)

    Returns:
        df itself

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st`

    """
    for index, row in df.iterrows():
        if 'State' not in list(row.index) or row['State'] in [None, np.nan, 'RUNNING']:
            try:
                jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
                seff = os.popen('seff ' + jobid)  # get job information
                for line in seff.readlines():
                    line = line.split(
                        ': ')  # must have space to be differentiated from time format 00:12:34
                    if len(line) == 2:
                        key, value = line
                        key = '_'.join(key.split(' '))  # change 'CPU utilized' to 'CPU_utilized'
                        value = value.split('\n')[0]
                        df.at[index, key] = value
            except:
                pass
    return df


def correct_type(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the type of values in `df`. to avoid the ID or other int valuables become float number.

        Args:
            df: A DataFrame saving metrics (and other super-parameters)

        Returns:
            df itself

        Examples:
            :func:`ssc_scoring.mymodules.tool.record_1st`

        """
    for column in df:
        ori_type = type(df[column].to_list()[-1])  # find the type of the last valuable in this column
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def get_df_id(record_file: str) -> Tuple[pd.DataFrame, int]:
    """Get the current experiment ID. It equals to the latest experiment ID + 1.

    Args:
        record_file: A file to record experiments details (super-parameters and metrics).

    Returns:
        dataframe and new_id

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st`

    """
    if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
        new_id = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(record_file)  # read the record file,
        last_id = df['ID'].to_list()[-1]  # find the last ID
        new_id = int(last_id) + 1
    return df, new_id


def record_1st(args: argparse.Namespace):
    """First record in this experiment.

    Args:
        task: 'score' or 'pos' for score and position prediction respectively.
        args: arguments.

    Returns:
        new_id

    Examples:
        :func:`ssc_scoring.run` and :func:`ssc_scoring.run_pos`

    """
    record_file = MypathBase().record_fpath
    lock = FileLock(str(record_file) + ".lock")  # lock the file, avoid other processes write other things
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a'):
            df, new_id = get_df_id(record_file)
            mypath = Mypath(new_id, check_id_dir=True)  # to check if id_dir already exist

            start_date = datetime.date.today().strftime("%Y-%m-%d")
            start_time = datetime.datetime.now().time().strftime("%H:%M:%S")
            # start record by id, date,time row = [new_id, date, time, ]
            idatime = {'ID': new_id, 'start_date': start_date, 'start_time': start_time}
            args_dict = vars(args)
            args_dict.update(idatime)

            if len(df) == 0:  # empty file
                df = pd.DataFrame([args_dict])  # need a [] , or need to assign the index for df
            else:
                index = df.index.to_list()[-1]  # last index
                for key, value in args_dict.items():  # write new line
                    if key=='ID':
                        df.at[index + 1, key] = value  #

            # df = fill_running(df)  # fill the state information for other experiments
            # df = correct_type(df)  # aviod annoying thing like: ID=1.00
            write_and_backup(df, record_file, mypath)
    return new_id, args_dict


def record_2nd(log_dict: dict, args: argparse.Namespace) -> None:
    """Second time to save logs.

    Args:
        task: 'score' or 'pos' for score and position prediction respectively.
        current_id: Current experiment ID
        log_dict: dict to save super-parameters and metrics
        args: arguments

    Returns:
        None. log_dict saved to disk.

    Examples:
        :func:`ssc_scoring.run` and :func:`ssc_scoring.run_pos`

    """
    current_id = args.id
    record_file = MypathBase().record_fpath

    lock = FileLock(str(record_file) + ".lock")
    with lock:  # with this lock,  open a file for exclusive access
        df = pd.read_csv(record_file)
        index = df.index[df['ID'] == current_id].to_list()
        if len(index) > 1:
            raise Exception("over 1 row has the same id", id)
        elif len(index) == 0:  # only one line,
            index = 0
        else:
            index = index[0]

        start_date = datetime.date.today().strftime("%Y-%m-%d")
        start_time = datetime.datetime.now().time().strftime("%H:%M:%S")
        df.at[index, 'end_date'] = start_date
        df.at[index, 'end_time'] = start_time

        # usage
        f = "%Y-%m-%d %H:%M:%S"
        t1 = datetime.datetime.strptime(df['start_date'][index] + ' ' + df['start_time'][index], f)
        t2 = datetime.datetime.strptime(df['end_date'][index] + ' ' + df['end_time'][index], f)
        elapsed_time = time_diff(t1, t2)
        df.at[index, 'elapsed_time'] = elapsed_time

        current_mypath = Mypath(current_id, check_id_dir=False)  # evaluate old model
        old_mypath = Mypath(args.infer_ID, check_id_dir=False)

        df = add_best_metrics(df, current_mypath, old_mypath, index)

        for key, value in log_dict.items():  # convert numpy to str before writing all log_dict to csv file
            if type(value) in [np.ndarray, list]:
                str_v = ''
                for v in value:
                    str_v += str(v)
                    str_v += '_'
                value = str_v
            df.loc[index, key] = value
            if type(value) is int:
                df[key] = df[key].astype('Int64')

        for column in df:
            if type(df[column].to_list()[-1]) is int:
                df[column] = df[column].astype('Int64')  # correct type again, avoid None/1.00/NAN, etc.

        args_dict = vars(args)
        args_dict.update({'ID': current_id})
        for column in df:
            if column in args_dict.keys() and type(args_dict[column]) is int:
                df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int
        write_and_backup(df, record_file, current_mypath)


def time_diff(t1: datetime, t2: datetime) -> str:
    """Time difference.

    Args:
        t1: time 1
        t2: time 2

    Returns:
        Elapsed time

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    # t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    # t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2 - t1

    return str(t_elapsed).split('.')[0]  # drop out microseconds


def _bytes_to_megabytes(value_bytes: int) -> float:
    """Convert bytes to megabytes.

    Args:
        value_bytes: bytes number

    Returns:
        megabytes

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_gpu_info`

    """
    return round((value_bytes / 1024) / 1024, 2)


def record_mem_info() -> int:
    """

    Returns:
        Memory usage in kB

    .. warning::

        This function is not tested. Please double check its code before using it.

    """

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    print('int(memusage.strip())')

    return int(memusage.strip())


def record_cgpu_info(outfile) -> Tuple:
    """Record GPU information to `outfile`.

    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        gpu_name, gpu_usage, gpu_util

    Examples:

        >>> record_gpu_info('slurm-98234.out')

        or

        :func:`ssc_scoring.run.gpu_info` and :func:`ssc_scoring.run_pos.gpu_info`

    """
    t = threading.currentThread()
    t.do_run = True

    if outfile:
        cpu_count = psutil.cpu_count()
        log_param('cpu_count', cpu_count)

        pid = os.getpid()
        python_process = psutil.Process(pid)

        jobid_gpuid = outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        log_param('gpuname', gpuname)
        # log_dict['gpuname'] = gpuname

        # log_dict['gpu_mem_usage'] = gpu_mem_usage
        # gpu_util = 0
        i = 0
        period = 2  # 2 seconds
        while i<60*20:  # stop signal passed from t, monitor 20 minutes
            if t.do_run:
                memoryUse = python_process.memory_info().rss / 2. ** 30  # memory use in GB...I think
                log_metric('cpu_mem_used_GB_in_process_rss', memoryUse, step=i)
                memoryUse = python_process.memory_info().vms / 2. ** 30  # memory use in GB...I think
                log_metric('cpu_mem_used_GB_in_process_vms', memoryUse, step=i)
                cpu_percent = psutil.cpu_percent()
                log_metric('cpu_util_used_percent', cpu_percent, step=i)
                # gpu_mem = dict(psutil.virtual_memory()._asdict())
                # log_params(gpu_mem)
                cpu_mem_used = psutil.virtual_memory().percent
                log_metric('cpu_mem_used_percent', cpu_mem_used, step=i)

                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # gpu_util += res.gpu
                log_metric("gpu_util", res.gpu, step=i)

                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                # gpu_mem_used = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total))
                gpu_mem_used = _bytes_to_megabytes(info.used)
                log_metric('gpu_mem_used_MB', gpu_mem_used, step=i)

                time.sleep(period)
                i += period
            else:
                print('record_cgpu_info do_run is True, let stop the process')
                break
        print('It is time to stop this process: record_cgpu_info')
        return None
        # gpu_util = gpu_util / 5
        # gpu_mem_usage = str(gpu_mem_used) + ' MB'

        # log_dict['gpu_util'] = str(gpu_util) + '%'
        # return gpuname, gpu_mem_usage, str(gpu_util) + '%'


    else:
        print('outfile is None, can not show GPU memory info')
        return None, None, None


def gpu_info(outfile: str) -> None:
    """Get GPU usage information.

    This function needs to be in the main file because it will be executed by another thread.

    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        None. The GPU information will be saved to global variable `log_dict`.

    Example:

    >>> gpu_info('slurm-98234.out')

    """
    gpu_name, gpu_usage, gpu_utis = record_cgpu_info(outfile)

    # log_dict['gpuname'], log_dict['gpu_mem_usage'], log_dict['gpu_util'] = gpu_name, gpu_usage, gpu_utis

    return None