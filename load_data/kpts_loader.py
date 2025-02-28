import os
import numpy as np
import pose
import json

def load_kpts_from_json(input_file_path: os.PathLike, extended = False) :
    """Given the path o a file it returns the input for HHP-Net

    Args:
        input_file_path: path of the json file contaings keypoints

    Returns:
        tensorflow tensor
    """
    with open(input_file_path) as f: #'/home/federico/Videos/01/key_points_2d_json/frame_00003_rgb.json'
        data = json.load(f)

    # load_data from file, select the first person detected
    my_kpts = pose.KeyPoints2D.from_json(data['people'][0]['pose_key_points_2d'][0])
    # take only the facial keypoints
    face_kpts = my_kpts.get_face_points() #change
    # normalise with respect the distance to the centroid
    face_kpts = face_kpts.get_normalised(confidence_threshold=-0.0001) # to take also with zero unc
    # transform to tensor
    #face_kpts_tf = face_kpts.get_tensor()
    if extended:
        face_kpts_np = face_kpts.get_numpy_extended()
    else:
        face_kpts_np = face_kpts.get_numpy()
    # face_kpts_np = np.asarray(face_kpts)
    
    return face_kpts_np


def load_face_kpts_from_json_as_tensor(input_file_path: os.PathLike, extended=False):
    """Given the path o a file it returns the input for HHP-Net

    Args:
        input_file_path: path of the json file contaings keypoints

    Returns:
        tensorflow tensor
    """
    with open(input_file_path) as f:  # '/home/federico/Videos/01/key_points_2d_json/frame_00003_rgb.json'
        data = json.load(f)

    # load_data from file, select the first person detected
    my_kpts = pose.KeyPoints2D.from_json(data['people'][0]['pose_key_points_2d'][0])
    # take only the facial keypoints
    face_kpts = my_kpts.get_face_points()
    # normalise with respect the distance to the centroid
    face_kpts = face_kpts.get_normalised()
    # transform to tensor
    # face_kpts_tf = face_kpts.get_tensor()
    if extended:
        raise NotImplementedError('extended version for tensor not available')
    else:
        face_kpts_np = face_kpts.get_tensor()
    # face_kpts_np = np.asarray(face_kpts)

    return face_kpts_np

if __name__ == '__main__':
    med = load_kpts_from_json('/media/DATA/Datasets/BIWI_processed/kpts_mediapipe/01/key_points_2d_json/frame_00004_rgb.json')
    #deleate?
    #change ro alpha
    ope = load_kpts_from_json(
        '/media/DATA/Datasets/BIWI_processed/kpts_openpose/01/key_points_2d_json/frame_00004_rgb.json')
    cent = load_kpts_from_json(
        '/media/DATA/Datasets/BIWI_processed/kpts_centernet_new/01/key_points_2d_json/frame_00004_rgb.json')
    print(f'centernet = {cent}\nopenpose = {ope}\nmediapipe = {med}')