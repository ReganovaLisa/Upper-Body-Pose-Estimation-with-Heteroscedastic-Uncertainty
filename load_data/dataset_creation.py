import os
from pathlib import Path

from numpy import ndarray
from exceptions import *

from load_data.gt_loader import load_ground_truth
from load_data.kpts_loader import load_kpts_from_json
import numpy as np


def calculate_stats_in_dataset(data):
    # extract the 2,5,8,11,14
    if len(data[0,:])==15:
        new_data = data[:, [2, 5, 8, 11, 14]] #change. check alphapose keypoints labels
    elif len(data[0,:])==33:
        new_data = data[:, [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]]
    else:
        raise NotImplementedError('number of keypoints not supported')
    mean = np.mean(new_data)
    var = np.std(new_data)
    return mean, var


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def load_data(kpts_path=None, angles_gt=None, server=True, all_data=True, extended=False) -> tuple[
    ndarray, ndarray, ndarray]:
    """It loads the dataset with groundtruth. It permits to choose which data to use as training and test.

    Args:
        server: bool, to decide paths for dataset and groundtruth
    Returns:
        tuple[list, list]: each returned list is a composed of many lists, as many as videos in the dataset
    """

    if (kpts_path is None) or (angles_gt is None):
        raise IncorrectInputValue

    list_kpts = []
    list_gt = []
    group_name = np.empty(0, dtype=int)
    # create a list of files to open
    for video_kpts in kpts_path.iterdir():
        video_name = video_kpts.stem
        kpts_current_folder = video_kpts / 'key_points_2d_json'
        gt_current_folder = angles_gt / video_name
        # paths for all files of one video
        kpts, gt = get_one_person_files_path(kpts_current_folder, gt_current_folder, all_data)
        # create a list with same length of data with the group/person name
        int_number = int(video_name)
        # print(int_number)
        group = np.full(len(kpts), int_number)
        group_name = np.concatenate((group_name, group), dtype=int)
        list_gt = list_gt + gt
        list_kpts = list_kpts + kpts

    # data_kpts, data_gt = np.empty((1,15), dtype=float), np.empty((3), dtype=float)
    data_kpts, data_gt = [], []
    for a, b in zip(list_kpts, list_gt):
        temp_a = load_kpts_from_json(a, extended=extended)
        temp_b, _ = load_ground_truth(b)  # temp_b, _ = load_ground_truth(b)
        # data_kpts = np.concatenate((data_kpts, temp_a), axis=0)
        # data_gt = np.concatenate((data_gt, temp_b), axis=0)
        data_kpts.append(temp_a)
        data_gt.append(temp_b)
    data_kpts = np.asarray(data_kpts).squeeze()
    data_gt = np.asarray(data_gt)

    # find and delete rows with NaNs
    nan_position_1 = np.argwhere(np.isnan(data_kpts))
    if len(nan_position_1) != 0:
        data_kpts = np.delete(data_kpts, [nan_position_1[:, 0]], axis=0)
        data_gt = np.delete(data_gt, [nan_position_1[:, 0]], axis=0)
        group_name = np.delete(group_name, [nan_position_1[:, 0]], axis=0)
        raise Exception('there are still NaNs values')

    nan_position_2 = np.argwhere(np.isnan(data_gt))
    if len(nan_position_2) != 0:
        data_kpts = np.delete(data_kpts, [nan_position_2[:, 0]], axis=0)
        data_gt = np.delete(data_gt, [nan_position_2[:, 0]], axis=0)
        group_name = np.delete(group_name, [nan_position_2[:, 0]], axis=0)
        raise Exception('there are still NaNs values')

    return data_kpts, data_gt, group_name


def get_one_person_files_path(kpts_current_folder: os.PathLike, gt_current_folder: os.PathLike, all_data: bool) -> \
tuple[list, list]:
    """

    Args:
        kpts_current_folder:
        gt_current_folder:
        all_data: all dataset or a subset used only for

    Returns:

    """
    kpts, gt = [], []
    counter = 16
    for current_kpts_file in kpts_current_folder.iterdir():
        if not all_data:
            counter = counter - 1
            if counter < 0: break
        current_kpts_name = current_kpts_file.stem
        current_gt_name = current_kpts_name.replace('rgb', 'pose.bin')
        current_gt_file = gt_current_folder / current_gt_name
        if current_gt_file.is_file() and current_kpts_file.is_file():
            kpts.append(current_kpts_file)
            gt.append(current_gt_file)
        else:
            print(f'skipped {current_kpts_file}')
        # print(current_gt_file)
        # print(current_kpts_file)
    return sorted(kpts), sorted(gt)


if __name__=='__main__':
    kpts_path = Path('/media/DATA/Datasets/BIWI_processed/') / 'kpts_centernet_new' #change
    print(f'kpts path ={kpts_path}')
    angles_gt = Path('/media/DATA/Datasets/BIWI/db_annotations') #change
    my_data, my_gt, groups = load_data(kpts_path=kpts_path, angles_gt=angles_gt,
                                       all_data=False, extended=False)
    m, std = calculate_stats_in_dataset(my_data)
    print(m)
    print(std)