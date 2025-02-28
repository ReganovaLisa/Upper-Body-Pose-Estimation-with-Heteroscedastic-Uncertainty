import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy import ndarray


def load_one_frame():
    address = Path('/home/federico/Downloads/frame_00004_pose.bin') #change
    a = load_ground_truth(address)
    return None

#change function bellow
def load_ground_truth(address: os.PathLike ) -> tuple[ndarray, ndarray]:
    """

    Args:
        address: os.PathLike with the complete file path of the gt file

    Returns:
        [pitch, yaw, roll], [x_pos, y_pos, z_pos] = the 3 angles and the 3D position in the 3D world

    """

    with open(address, mode='rb') as file:
        content = file.read()
    float_dim_in_bytes = 4
    triplets = 3*float_dim_in_bytes
    x_pos, y_pos, z_pos = struct.unpack('fff', content[0:triplets])
    pitch, yaw, roll = struct.unpack('fff', content[triplets:2*triplets])
    return np.asarray([yaw, -pitch, roll], dtype=float), np.asarray([x_pos, y_pos, z_pos], dtype=float)

if __name__ == '__main__':
    load_one_frame()