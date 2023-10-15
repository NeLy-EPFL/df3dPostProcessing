from df3dPostProcessing import df3dPostProcess
from df3dPostProcessing.df3dPostProcessing import prism_skeleton_LP3D, flytracker_skel, df3d_skeleton
from df3dPostProcessing.utils import utils_plots


from pathlib import Path
import pickle
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

def load_csv(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    df = df.drop(columns=["frame_idx", "video_id"])

    joints_todf3d = {"Coxa":"ThC", "Femur":"CTr", "Tibia":"FTi", "Tarsus":"TiTa", "Claw":"Claw"}

    df3d_output_dict = {}

    assert len(df.columns)%3 == 0, "Number of columns in csv file is not a multiple of 3 (can not be xyz)"
    points_3d = np.zeros((len(df), len(flytracker_skel), 3))

    thorax_ref = df[["Th_x", "Th_y", "Th_z"]].values

    for i, joint in enumerate(flytracker_skel):
        seg = joint[:2]

        joint_ikjrec = joints_todf3d[joint[2:]]


        points_3d[:, i, 0] = df[f"{seg}-{joint_ikjrec}_x"]
        points_3d[:, i, 1] = df[f"{seg}-{joint_ikjrec}_y"]
        points_3d[:, i, 2] = df[f"{seg}-{joint_ikjrec}_z"]

    df3d_output_dict["points3d"] = points_3d.copy()
    return points_3d


base_path = Path("data/clean_3d_best_ventral_best_side.csv")
skel = "flytracker"
data = load_csv(base_path)

# save to pickle 
exp_path = base_path.parent / "df3dpostprocess_reshape.pkl"
with open(exp_path, "wb") as f:
    pickle.dump(data, f)
exp = str(exp_path)

# Read pose results and calculate 3d positions from 2d estimations
df3dpp = df3dPostProcess(exp, calculate_3d=False, skeleton=skel)


interpolate = False
# Align and scale 3d positions using the NeuroMechFly skeleton as template, data is interpolated 
align = df3dpp.align_to_template(interpolate=interpolate, original_time_step=1/60, window_length=10)

# Calculate joint angles from the leg (ThC-3DOF, CTr-2DOF, FTi-1DOF, TiTa-1DOF)
#df3dpp.skeleton = "flytracker"
angles = df3dpp.calculate_leg_angles()

# Can be used to  plot or show the 3d and inverse kinematics results
beg, end = 0, 31
if base_path.suffix == ".pkl":
    beg, end = 300, 400

 
show = False
utils_plots.plot_legs_from_angles(
        angles = angles,
        data_dict= align,
        exp_dir = str(base_path.parent),
        begin=beg,
        end=end,
        plane='xy',
        saveImgs=not show,
        dir_name='km',
        extraDOF={'CTr_roll':angles},
        ik_angles=False,
        pause=show,
        lim_axes=False)

utils_plots.plot_legs_from_angles(
        angles = angles,
        data_dict= align,
        exp_dir = str(base_path.parent),
        begin=beg,
        end=end,
        plane='xz',
        saveImgs=not show,
        dir_name='km',
        extraDOF={'CTr_roll':angles},
        ik_angles=False,
        pause=show,
        lim_axes=False)
