"""Plot the 3D figure with the pose in the space."""
from __future__ import annotations

import logging
import math
from math import cos
from math import sin
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import src.pose as pose
from src.utils import IMAGE_EXTENSION

logger = logging.getLogger(__name__)


def configure_plots_3d(height: int, width: int) -> tuple[Figure, Axes3D]:
    """Configure plots.

    Returns:
        matplotlib figure and axis
    """
    matplotlib.use("Agg")  # non gui backend
    px = 1 / 100  # dpi
    fig = plt.figure(figsize=(height * px, width * px))
    ax: Axes3D = plt.axes(projection="3d")
    fig.add_axes(ax)
    ax.view_init(azim=90, elev=115, vertical_axis="z")
    ax.set_xlim(-200.0, 400.0)
    ax.set_ylim(-300.0, 800.0)
    ax.set_zlim3d(800, 2200.0)
    # ax.set_xlim([0, 1280])
    # ax.set_ylim([0, 720])
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    return fig, ax


def draw_axis_3d(yaw, pitch, roll, ax=None, tdx=None, tdy=None, tdz=None, size=100):
    """Draw yaw pitch and roll axis on a 3D matplotlib environment.

    Args:
        yaw (float): value that represents the yaw rotation of the face
        pitch (float): value that represents the pitch rotation of the face
        roll (float): value that represents the roll rotation of the face
        ax (axis of matplotlib): matplotlib axis obj to draw on
            (default is None)
        tdx (float64): x coordinate from where the vector drawing
            start expressed in pixel coordinates
            (default is None)
        tdy (float64): y coordinate from where the vector drawing
            start expressed in pixel coordinates
            (default is None)
        tdz (float64): x coordinate from where the vector drawing
            start expressed in pixel coordinates
            (default is None)
        size (int): value that will be multiplied to each x, y and z
            value that enlarge the "vector drawing"
            (default is 50)

    Returns:
        list_points_vector (list): list containing the vectors
            [x, y, z]= [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if (tdx is None) and (tdy is None) and (tdz is None):
        height, width = 720, 1280
        tdx = width / 2
        tdy = height / 2
        tdz = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    z1 = size * (sin(pitch) * sin(roll) - cos(pitch) * cos(roll) * sin(yaw)) + tdz

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    z2 = size * (cos(roll) * sin(pitch) + cos(pitch) * sin(yaw) * sin(roll)) + tdz

    # Z-Axis (out of the screen) drawn in yellow (previously blue)
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    z3 = size * (cos(pitch) * cos(yaw)) + tdz

    try:
        ax.plot(
            [int(tdx), int(x1)],
            [int(tdy), int(y1)],
            [int(tdz), int(z1)],
            c="red",
        )
        ax.plot(
            [int(tdx), int(x2)],
            [int(tdy), int(y2)],
            [int(tdz), int(z2)],
            c="green",
        )
        ax.plot(
            [int(tdx), int(x3)],
            [int(tdy), int(y3)],
            [int(tdz), int(z3)],
            c="yellow",
        )
    except ValueError as e:
        return ValueError(str(e))

    list_points_vector = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
    return list_points_vector


def plot_velocity(
    x_pos: pd.Series,
    y_pos: pd.Series,
    z_pos: pd.Series,
    speed: pd.Series,
    joint_name: str,
    folder: Path,
):
    """Plot velocity (x,y,z,magnitude) time series.

    Args:
        x_pos: x position time series
        y_pos: y position time series
        z_pos: z position time series
        speed: speed time series
        joint_name: name of joint
        folder: folder where plots will be saved

    Returns:
        None

    Notes:
        Legend positions
            - 'best'            0
            - 'upper right'     1
            - 'upper left'      2
            - 'lower left'      3
            - 'lower right'     4
            - 'right'           5
            - 'center left'     6
            - 'center right'    7
            - 'lower center'    8
            - 'upper center'    9
            - 'center'          10
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex="row")  # , sharey=True
    fig.subplots_adjust(hspace=0.5)

    ax1.plot(x_pos, label="x")
    ax1.set_xlabel("time [frames]")
    ax1.set_ylabel("[pixel or m/s]")
    ax1.legend(prop={"size": 15}, loc="upper left")

    ax2.plot(y_pos, label="y")
    ax2.set_xlabel("time [frames]")
    ax2.set_ylabel("[pixel or m/s]")
    ax2.legend(prop={"size": 15}, loc="upper left")

    ax3.plot(z_pos, label="z")
    ax3.set_xlabel("time [frames]")
    ax3.set_ylabel("[pixel or m/s]")
    ax3.legend(prop={"size": 15}, loc="upper left")

    ax4.plot(speed, label="speed")
    ax4.set_xlabel("time [frames]")
    ax4.set_ylabel("[pixel or m/s]")
    ax4.legend(prop={"size": 15}, loc="upper left")

    name_fig = folder / f"{joint_name}{IMAGE_EXTENSION}"
    logger.debug("Saving fig at %s", name_fig)
    fig.savefig(str(name_fig), dpi=500)


def draw_3d_scene(
    fig: Figure,
    ax: Axes3D,
    key_points_3d: pose.KeyPoints3D,
    head_pose: pose.HeadPose,
) -> tuple[Figure, Axes3D]:
    """The function plots each pose in the scene and head directions.

    Args:
        fig: matplotlib figure
        ax: ax where to draw
        key_points_3d_list: list of key points per people
        path_ypr (Path or string): path of the folder containing
            json head direction data

    Returns:
        [type]: [description]
    """
    ax = draw_key_points_pose(key_points=key_points_3d, image=None, ax=ax)

    if key_points_3d.model_type is not pose.ModelType.MEDIAPIPEHANDS:
        ret = draw_axis_3d(
            head_pose.yaw,
            head_pose.pitch,
            head_pose.roll,
            ax=ax,
            tdx=key_points_3d[pose.Joints.NOSE].x,
            tdy=key_points_3d[pose.Joints.NOSE].y,
            tdz=key_points_3d[pose.Joints.NOSE].z,
        )
        match ret:
            case ValueError():
                logger.error("Cannot draw head axis. Values were nan.")

    return fig, ax


def matplotlib_canvas_to_rgb_array(fig: Figure) -> np.ndarray:
    """Draw a figure and return its RGB24 array.

    Args:
        fig: figure

    Returns:
        RGB24 numpy array.
    """
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def draw_key_points_pose(
    key_points: pose.KeyPoints2D | pose.KeyPoints3D,
    image: np.ndarray = None,
    ax: Axes3D = None,
    low_confidence_threshold: float = 0.15,
) -> np.ndarray | Axes3D:
    """Draw the key points and the links connecting them.

    Args:
        ax:
        image (numpy.ndarray): The image where the lines connecting
            the key points will be printed
        key_points (KeyPoints2D | KeyPoints3D): all detected key points
        low_confidence_threshold (float): threshold below which
            key points will be plotted with LOW_CONFIDENCE colour

    Returns:
        img (numpy.ndarray): The image with the drawings of lines and key points
    """
    if (image is not None) and (key_points.key_point_type is pose.KeyPoint2D):
        overlay = image.copy()
        dim_2d = True
        dim_3d = False
    elif (key_points is not None) and (key_points.key_point_type is pose.KeyPoint3D):
        dim_2d = False
        dim_3d = True
    else:
        raise ValueError("Key points type and input do not match.")

    for joint, kp in key_points.items():
        if math.isclose(kp.confidence, 0, abs_tol=low_confidence_threshold):
            color = COLOR_POSE[LOW_CONFIDENCE]
        elif joint in pose.LEFT_HAND_JOINTS:
            color = COLOR_POSE[LEFT_HAND]
        elif joint in pose.RIGHT_HAND_JOINTS:
            color = COLOR_POSE[RIGHT_HAND]
        else:
            color = COLOR_POSE.get(joint, COLOR_POSE[DEFAULT])

        if dim_2d:
            cv2.circle(image, (int(kp.x), int(kp.y)), 1, color, 2)
        elif dim_3d:
            ax.scatter3D(kp.x, kp.y, kp.z, c=np.fliplr(np.array([color]) / 255))  # RGB

    for joint_a, joint_b in key_points.get_body_links():
        if (joint_a in key_points) & (joint_b in key_points):
            if dim_2d:
                cv2.line(
                    overlay,
                    (int(key_points[joint_a].x), int(key_points[joint_a].y)),
                    (int(key_points[joint_b].x), int(key_points[joint_b].y)),
                    COLOR_POSE[LINK],
                    2,
                )
            elif dim_3d:
                ax.plot(
                    (key_points[joint_a].x, key_points[joint_b].x),
                    (key_points[joint_a].y, key_points[joint_b].y),
                    (key_points[joint_a].z, key_points[joint_b].z),
                    c=np.fliplr(np.array([(0, 0, 0)]) / 255),  # RGB
                )

    alpha = 0.4
    if dim_2d:
        image_w = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return image_w
    elif dim_3d:
        return ax


def draw_axis_2d(
    yaw: float,
    pitch: float,
    roll: float,
    image: np.ndarray = None,
    tdx: np.float64 = None,
    tdy: np.float64 = None,
    size: int = 100,
):
    """Draw YPR axis and return the vector project on the image plane.

    Args:
        :yaw (float): value that represents the yaw rotation of the face
        :pitch (float): value that represents the pitch rotation of the face
        :roll (float): value that represents the roll rotation of the face
        :image (numpy.ndarray): The image where the three vector will be printed
            (default is None)
        :tdx (float64): x coordinate from where the vector drawing
            start expressed in pixel coordinates
            (default is None)
        :tdy (float64): y coordinate from where the vector drawing
            start expressed in pixel coordinates
            (default is None)
        :size (int): value that will be multiplied to each x, y and z
            value that enlarge the "vector drawing"
            (default is 50)

    Returns:
        :list_projection_xy (list): list containing the unit vector [x, y, z]
    """
    pitch, roll, yaw = ypr_deg2rad(pitch, roll, yaw)
    tdx, tdy = td(image, tdx, tdy)
    x, y = project_3d_to_2d_xy(pitch, roll, size, tdx, tdy, yaw)

    if image is not None:
        cv2.line(image, (int(tdx), int(tdy)), (int(x[0]), int(y[0])), (0, 0, 255), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x[1]), int(y[1])), (0, 255, 0), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x[2]), int(y[2])), (255, 0, 0), 2)

    list_projection_xy = [math.sin(yaw), -math.cos(yaw) * math.sin(pitch)]
    return list_projection_xy


def project_3d_to_2d_xy(pitch: float, roll: float, size: int, tdx, tdy, yaw):
    """Project from 3D to 2D XY plane (Z = 0).

    Args:
        pitch:
        roll:
        size:
        tdx:
        tdy:
        yaw:

    Notes:
        z3 = size * (math.cos(pitch) * math.cos(yaw)) + tdy

    """
    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = (
        size
        * (
            math.cos(pitch) * math.sin(roll)
            + math.cos(roll) * math.sin(pitch) * math.sin(yaw)
        )
        + tdy
    )
    # Y-Axis | drawn in green
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = (
        size
        * (
            math.cos(pitch) * math.cos(roll)
            - math.sin(pitch) * math.sin(yaw) * math.sin(roll)
        )
        + tdy
    )

    # Z-Axis (out of the screen) drawn in yellow (previously blue)
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    x = [x1, x2, x3]
    y = [y1, y2, y3]

    return x, y


def td(image, tdx, tdy):
    """Default values for tdx and tdy.

    Args:
        image: image
        tdx: input tdx
        tdy: input tdy

    Returns:
        tdx and tdy in that order.
    """
    if (tdx is None) and (tdy is None):
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2

    return tdx, tdy


def ypr_deg2rad(pitch, roll, yaw):
    """Convert YPR values to radians.

    Args:
        pitch: degrees value
        roll: degrees value
        yaw: degrees value

    Returns:
        pitch, roll and yaw in that order.
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    return pitch, roll, yaw


DEFAULT = "default"
LOW_CONFIDENCE = "low_confidence"
LINK = "link"
RIGHT_HAND = "right_hand"
LEFT_HAND = "left_hand"
COLOR_POSE = {
    pose.Joints.NOSE: (255, 0, 100),  # purple
    pose.Joints.L_EYE: (0, 255, 0),  # green
    pose.Joints.R_EYE: (220, 0, 255),  # dark_pink
    pose.Joints.L_EAR: (0, 80, 255),  # light_orange
    pose.Joints.R_EAR: (0, 220, 255),  # yellow
    DEFAULT: (255, 0, 0),  # blue
    LOW_CONFIDENCE: (0, 0, 0),  # black
    LINK: (255, 255, 255),  # white
    LEFT_HAND: (0, 0, 255),  # red
    RIGHT_HAND: (255, 255, 0),  # cyan
}
