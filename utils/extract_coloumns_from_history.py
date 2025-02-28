import pathlib
from pathlib import Path

import numpy as np
import pandas as pd


def extract_and_join(folder: pathlib):
    val_yaw = []
    val_pitch = []
    val_roll = []
    folder_to_save = folder / 'stats'
    Path(folder_to_save).mkdir(parents=True, exist_ok=True)
    for f in folder.iterdir():
        if f.is_file() and f.suffix=='.csv':
            temp = pd.read_csv(f)
            # tr = temp[['yaw_Save_UNC', 'pitch_Save_UNC', 'roll_Save_UNC']]
            try:
                val_yaw.append(temp['val_yaw_Save_UNC'])
                val_pitch.append(temp['val_pitch_Save_UNC'])
                val_roll.append(temp['val_roll_Save_UNC'])
            except KeyError:
                print(f'{f} has no key value for this')
                continue
    val_yaw = np.asarray(val_yaw)
    val_pitch = np.asarray(val_pitch)
    val_roll = np.asarray(val_roll)
    pd.DataFrame(val_yaw).to_csv(str(folder_to_save / 'uncertainty_val_history_YAW.csv'), index=False, header=False,
                                 float_format='%.2f')
    pd.DataFrame(val_pitch).to_csv(str(folder_to_save / 'uncertainty_val_history_PITCH.csv'), index=False, header=False,
                                   float_format='%.2f')
    pd.DataFrame(val_roll).to_csv(str(folder_to_save / 'uncertainty_val_history_ROLL.csv'), index=False, header=False,
                                  float_format='%.2f')
    # fig_name = folder / 'uncertainty_training_hist.png'
    # ax = tr.plot(figsize=(12, 4), subplots=True)
    # ax.figure.savefig(fig_name)
    # fig_name = folder/'uncertainty_validation_hist.png'
    # ax1 = val.plot()
    # ax1.figure.savefig(fig_name)

    return 0


if __name__=='__main__':
    train_path = Path('/media/DATA/Users/Federico/Tensorboard/HPE-Net/15-06-2022_16_52_OPENPOSE_LOO')
    extract_and_join(train_path)