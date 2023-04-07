import numpy as np
import os
import pickle
import scipy.signal
from pathlib import Path
from .utils.utils_alignment import align_3d
from .utils.utils_angles import calculate_angles
from .utils.utils_ball_info import ball_size_and_pos
from .utils.utils_outliers import correct_outliers
#from .utils.utils_plots import *

df3d_skeleton = ['RFCoxa',
                 'RFFemur',
                 'RFTibia',
                 'RFTarsus',
                 'RFClaw',
                 'RMCoxa',
                 'RMFemur',
                 'RMTibia',
                 'RMTarsus',
                 'RMClaw',
                 'RHCoxa',
                 'RHFemur',
                 'RHTibia',
                 'RHTarsus',
                 'RHClaw',
                 'RAntenna',
                 'RStripe1',
                 'RStripe2',
                 'RStripe3',
                 'LFCoxa',
                 'LFFemur',
                 'LFTibia',
                 'LFTarsus',
                 'LFClaw',
                 'LMCoxa',
                 'LMFemur',
                 'LMTibia',
                 'LMTarsus',
                 'LMClaw',
                 'LHCoxa',
                 'LHFemur',
                 'LHTibia',
                 'LHTarsus',
                 'LHClaw',
                 'LAntenna',
                 'LStripe1',
                 'LStripe2',
                 'LStripe3']

prism_skeleton_LP3D = ['LFCoxa',
                       'LFFemur',
                       'LFTibia',
                       'LFTarsus',
                       'LFClaw',
                       'LMCoxa',
                       'LMFemur',
                       'LMTibia',
                       'LMTarsus',
                       'LMClaw',
                       'LHCoxa',
                       'LHFemur',
                       'LHTibia',
                       'LHTarsus',
                       'LHClaw',
                       'RFCoxa',
                       'RFFemur',
                       'RFTibia',
                       'RFTarsus',
                       'RFClaw',
                       'RMCoxa',
                       'RMFemur',
                       'RMTibia',
                       'RMTarsus',
                       'RMClaw',
                       'RHCoxa',
                       'RHFemur',
                       'RHTibia',
                       'RHTarsus',
                       'RHClaw']

prism_skeleton = ['RFCoxa',
                  'RFFemur',
                  'RFTibia',
                  'RFTarsus',
                  'RFClaw',
                  'RMCoxa',
                  'RMFemur',
                  'RMTibia',
                  'RMTarsus',
                  'RMClaw',
                  'RHCoxa',
                  'RHFemur',
                  'RHTibia',
                  'RHTarsus',
                  'RHClaw',
                  'LFCoxa',
                  'LFFemur',
                  'LFTibia',
                  'LFTarsus',
                  'LFClaw',
                  'LMCoxa',
                  'LMFemur',
                  'LMTibia',
                  'LMTarsus',
                  'LMClaw',
                  'LHCoxa',
                  'LHFemur',
                  'LHTibia',
                  'LHTarsus',
                  'LHClaw',
                  'RAntenna',
                  'LAntenna',
                  'REye',
                  'LEye',
                  'RHaltere',
                  'LHaltere',
                  'RWing',
                  'LWing',
                  'Proboscis',
                  'Neck',
                  'Genitalia',
                  'Scutellum',
                  'A1A2',
                  'A3',
                  'A4',
                  'A5',
                  'A6']

# A. Ari Addition:

flytracker_skel = ['RFCoxa',
                 'RFFemur',
                 'RFTibia',
                 'RFTarsus',
                 'RFClaw',
                 'RMCoxa',
                 'RMFemur',
                 'RMTibia',
                 'RMTarsus',
                 'RMClaw',
                 'RHCoxa',
                 'RHFemur',
                 'RHTibia',
                 'RHTarsus',
                 'RHClaw',
                 'LFCoxa',
                 'LFFemur',
                 'LFTibia',
                 'LFTarsus',
                 'LFClaw',
                 'LMCoxa',
                 'LMFemur',
                 'LMTibia',
                 'LMTarsus',
                 'LMClaw',
                 'LHCoxa',
                 'LHFemur',
                 'LHTibia',
                 'LHTarsus',
                 'LHClaw']

template_nmf = {'RFCoxa': [0.35, -0.27, 0.400],
                'RFFemur': [0.35, -0.27, -0.025],
                'RFTibia': [0.35, -0.27, -0.731],
                'RFTarsus': [0.35, -0.27, -1.249],
                'RFClaw': [0.35, -0.27, -1.912],
                'LFCoxa': [0.35, 0.27, 0.400],
                'LFFemur': [0.35, 0.27, -0.025],
                'LFTibia': [0.35, 0.27, -0.731],
                'LFTarsus': [0.35, 0.27, -1.249],
                'LFClaw': [0.35, 0.27, -1.912],
                'RMCoxa': [0, -0.125, 0],
                'RMFemur': [0, -0.125, -0.182],
                'RMTibia': [0, -0.125, -0.965],
                'RMTarsus': [0, -0.125, -1.633],
                'RMClaw': [0, -0.125, -2.328],
                'LMCoxa': [0, 0.125, 0],
                'LMFemur': [0, 0.125, -0.182],
                'LMTibia': [0, 0.125, -0.965],
                'LMTarsus': [0, 0.125, -1.633],
                'LMClaw': [0, 0.125, -2.328],
                'RHCoxa': [-0.215, -0.087, -0.073],
                'RHFemur': [-0.215, -0.087, -0.272],
                'RHTibia': [-0.215, -0.087, -1.108],
                'RHTarsus': [-0.215, -0.087, -1.793],
                'RHClaw': [-0.215, -0.087, -2.588],
                'LHCoxa': [-0.215, 0.087, -0.073],
                'LHFemur': [-0.215, 0.087, -0.272],
                'LHTibia': [-0.215, 0.087, -1.108],
                'LHTarsus': [-0.215, 0.087, -1.793],
                'LHClaw': [-0.215, 0.087, -2.588],
                'RAntenna': [0.25, -0.068, 0.67],
                'LAntenna': [0.25, 0.068, 0.67]}

zero_pose_nmf = {'RF_leg': {'ThC_yaw': 0,
                            'ThC_pitch': 0,
                            'ThC_roll': 0,
                            'CTr_roll': 0,
                            'CTr_pitch': -np.pi,
                            'FTi_pitch': np.pi,
                            'TiTa_pitch': -np.pi},
                 'RM_leg': {'ThC_yaw': 0,
                            'ThC_pitch': 0,
                            'ThC_roll': -np.pi / 2,
                            'CTr_roll': 0,
                            'CTr_pitch': -np.pi,
                            'FTi_pitch': np.pi,
                            'TiTa_pitch': -np.pi},
                 'RH_leg': {'ThC_yaw': 0,
                            'ThC_pitch': 0,
                            'ThC_roll': -np.pi,
                            'CTr_roll': 0,
                            'CTr_pitch': -np.pi,
                            'FTi_pitch': np.pi,
                            'TiTa_pitch': -np.pi},
                 'LF_leg': {'ThC_yaw': 0,
                            'ThC_pitch': 0,
                            'ThC_roll': 0,
                            'CTr_roll': 0,
                            'CTr_pitch': -np.pi,
                            'FTi_pitch': np.pi,
                            'TiTa_pitch': -np.pi},
                 'LM_leg': {'ThC_yaw': 0,
                            'ThC_pitch': 0,
                            'ThC_roll': np.pi / 2,
                            'CTr_roll': 0,
                            'CTr_pitch': -np.pi,
                            'FTi_pitch': np.pi,
                            'TiTa_pitch': -np.pi},
                 'LH_leg': {'ThC_yaw': 0,
                            'ThC_pitch': 0,
                            'ThC_roll': np.pi,
                            'CTr_roll': 0,
                            'CTr_pitch': -np.pi,
                            'FTi_pitch': np.pi,
                            'TiTa_pitch': -np.pi}}


class df3dPostProcess:
    def __init__(
            self,
            exp_dir,
            multiple=False,
            file_name='',
            skeleton='df3d',
            calculate_3d=False,
            outlier_correction=False):
        self.exp_dir = exp_dir
        self.raw_data_3d = np.array([])
        self.raw_data_2d = np.array([])
        self.skeleton = skeleton
        self.template = template_nmf
        self.zero_pose = zero_pose_nmf
        self.raw_data_cams = {}
        self.load_data(
            exp_dir,
            calculate_3d,
            skeleton,
            multiple,
            file_name,
            outlier_correction=outlier_correction)
        self.data_3d_dict = load_data_to_dict(self.raw_data_3d, skeleton)
        self.data_2d_dict = load_data_to_dict(self.raw_data_2d, skeleton)
        self.aligned_model = {}

    def align_to_template(
            self,
            scale='local',
            all_body=False,
            interpolate=False,
            smoothing=True,
            original_time_step=0.01,
            new_time_step=0.001,
            window_length=29):
        self.aligned_model = align_3d(
            self.data_3d_dict,
            self.skeleton,
            self.template,
            scale,
            all_body,
            interpolate,
            smoothing,
            original_time_step,
            new_time_step,
            window_length)

        return self.aligned_model

    def calculate_leg_angles(
            self,
            begin=0,
            end=0,
            get_CTr_roll=True,
            save_angles=False,
            use_zero_pose=True,
            zero_pose=None):
        if not zero_pose:
            zero_pose = self.zero_pose

        leg_angles = calculate_angles(
            self.aligned_model, begin, end, get_CTr_roll, zero_pose)

        if use_zero_pose:
            for leg, angles in leg_angles.items():
                for angle, vals in angles.items():
                    leg_angles[leg][angle] = list(
                        np.array(vals) + zero_pose[leg][angle])

        if save_angles:
            path = 'joint_angles.pkl'
            if self.skeleton == 'df3d':
                path = self.exp_dir.replace('pose_result', 'joint_angles')
            if self.skeleton == 'prism' or self.skeleton == 'lp3d'or self.skeleton == 'flytracker': # Ari. FT addition
                folders = self.exp_dir.split('/')
                parent = self.exp_dir[:self.exp_dir.find(folders[-1])]
                path = os.path.join(parent, f"joint_angles__{folders[-1]}")
                path = path.replace(".npy", ".pkl")

            with open(path, 'wb') as f:
                pickle.dump(leg_angles, f)
        return leg_angles

    def calculate_velocity(
            self,
            data,
            window=11,
            order=3,
            time_step=0.001,
            save_velocities=False):
        velocities = {}
        for leg, angles in data.items():
            velocities[leg] = {}
            for th, data in angles.items():
                vel = scipy.signal.savgol_filter(
                    data, window, order, deriv=1, delta=time_step, mode='nearest')
                velocities[leg][th] = vel

        if save_velocities:
            path = 'joint_velocities.pkl'
            if self.skeleton == 'df3d':
                path = self.exp_dir.replace('pose_result', 'joint_velocities')
            if self.skeleton == 'prism' or self.skeleton == 'lp3d'or self.skeleton == 'flytracker': # Ari. FT addition
                folders = self.exp_dir.split('/')
                parent = self.exp_dir[:self.exp_dir.find(folders[-1])]
                path = os.path.join(parent, f"joint_velocities__{folders[-1]}")
                path = path.replace(".npy", ".pkl")

            with open(path, 'wb') as f:
                pickle.dump(velocities, f)

        return velocities

    def get_treadmill_info(
            self,
            path=None,
            save_ball_info=False,
            show_detection=False):
        if not path:
            data_path = self.exp_dir
        else:
            data_path = path

        ball_radius, ball_pos = ball_size_and_pos(data_path, show_detection)

        ball_info = {'radius': ball_radius, 'position': ball_pos}

        if save_ball_info:
            path = self.exp_dir.replace('pose_result', 'treadmill_info')
            with open(path, 'wb') as f:
                pickle.dump(ball_info, f)

        return ball_info

    def load_data(
            self,
            exp,
            calculate_3d,
            skeleton,
            multiple,
            file_name,
            outlier_correction):
        if multiple:
            currentDirectory = os.getcwd()
            dataFile = os.path.join(currentDirectory, file_name)
            data = np.load(dataFile, allow_pickle=True)
            self.raw_data_3d = data[exp]
        else:
            data = np.load(exp, allow_pickle=True)
            if skeleton == 'prism' or skeleton == 'lp3d' or skeleton == 'flytracker': # Ari. FT addition
                data = {'points3d_wo_procrustes': data}
            if calculate_3d:
                self.raw_data_3d = triangulate_2d(
                    data, exp, outlier_correction)

            for key, vals in data.items():
                if not isinstance(key, str):
                    self.raw_data_cams[key] = vals
                elif key == 'points3d_wo_procrustes':
                    if not calculate_3d:
                        self.raw_data_3d = vals
                elif key == 'points2d':
                    self.raw_data_2d = vals


def triangulate_2d(data, exp_dir, outlier_correction):
    img_folder = exp_dir[:exp_dir.find('df3d')]
    out_folder = exp_dir[:exp_dir.find('pose_result')]
    num_images = data['points3d'].shape[0]

    from deepfly.CameraNetwork import CameraNetwork
    camNet = CameraNetwork(
        image_folder=img_folder,
        output_folder=out_folder,
        num_images=num_images)

    for cam_id in range(7):
        camNet.cam_list[cam_id].set_intrinsic(data[cam_id]["intr"])
        camNet.cam_list[cam_id].set_R(data[cam_id]["R"])
        camNet.cam_list[cam_id].set_tvec(data[cam_id]["tvec"])
    camNet.triangulate()

    if outlier_correction:
        camNet = correct_outliers(camNet)

    return camNet.points3d


def load_data_to_dict(data, skeleton):
    final_dict = {}
    if len(data.shape) == 3:
        time_pts, body_parts, axes = data.shape
        num_cams = 1
    elif len(data.shape) == 4:
        num_cams, time_pts, body_parts, axes = data.shape
        cams_dict = {}
    else:
        return final_dict

    if skeleton == 'prism':
        tracked_joints = prism_skeleton
    elif skeleton == 'df3d':
        tracked_joints = df3d_skeleton
    elif skeleton == 'lp3d':
        tracked_joints = prism_skeleton_LP3D
    elif skeleton == 'flytracker':
        tracked_joints = flytracker_skel

    if body_parts != len(tracked_joints):
        raise Exception("Check tracked joints definition")

    for cam in range(num_cams):
        exp_dict = {}
        if num_cams > 1:
            coords = data[cam]
        else:
            coords = data
        for i, name in enumerate(tracked_joints):
            if 'Coxa' in name or 'Femur' in name or 'Tibia' in name or 'Tarsus' in name or 'Claw' in name:
                body_part = name[0:2] + '_leg'
                landmark = name[2:]
            else:
                if 'Antenna' in name or 'Neck' in name or 'Proboscis' in name or 'Eye' in name:
                    body_part = 'Head'
                elif 'Haltere' in name or 'Wing' in name or 'Scutellum' in name:
                    body_part = 'Thorax'
                elif 'Stripe' in name or 'Genitalia' in name or 'A1A2' in name or 'A3' in name or 'A4' in name or 'A5' in name or 'A6' in name:
                    body_part = 'Abdomen'
                landmark = name

            if body_part not in exp_dict.keys():
                exp_dict[body_part] = {}

            exp_dict[body_part][landmark] = coords[:, i, :]

        if num_cams > 1:
            cams_dict[cam] = exp_dict

    if num_cams > 1:
        final_dict = cams_dict
    else:
        final_dict = exp_dict

    return final_dict
