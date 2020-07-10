import numpy as np
import os
from .utils.utils_alignment import align_data, rescale_using_2d_data
from .utils.utils_angles import calculate_angles
from .utils.utils_plots import *

tracked_joints = ['LFCoxa',
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
                  'LStripe3',
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
                  'RHClaw',
                  'RAntenna',
                  'RStripe1',
                  'RStripe2',
                  'RStripe3']

class df3dPostProcess:
    def __init__(self, results_dir, multiple = False, file_name = ''):
        self.res_dir = results_dir
        self.tracked_joints = tracked_joints
        self.raw_data_3d = {}
        self.raw_data_2d = {}
        self.raw_data_cams = {}
        self.load_data(results_dir, multiple, file_name)
        self.data_3d_dict = load_data_to_dict(self.raw_data_3d)
        self.data_2d_dict = load_data_to_dict(self.raw_data_2d)
        self.aligned_model = {}

    def align_3d_data(self, rescale = True):
        self.aligned_model = align_data(self.data_3d_dict)
        if rescale:
            self.aligned_model = rescale_using_2d_data(self.aligned_model, self.data_2d_dict, self.raw_data_cams, self.res_dir)
            
        return self.aligned_model

    def calculate_leg_angles(self, begin = 0, end = 0, get_roll_tr = True):
        leg_angles = calculate_angles(self.aligned_model, begin, end, get_roll_tr)
        return leg_angles

    
    def load_data(self, exp, multiple = False, file_name = ''):
        if multiple:
            currentDirectory = os.getcwd()
            dataFile = os.path.join(currentDirectory,file_name)
            data = np.load(dataFile,allow_pickle=True)
            self.raw_data_3d  = data[exp]
        else:
            data = np.load(exp,allow_pickle=True)
            for key, vals in data.items():
                if not isinstance(key,str):
                    self.raw_data_cams[key] = vals
                elif key == 'points3d':
                    self.raw_data_3d = vals
                elif key == 'points2d':
                    self.raw_data_2d = vals       

def load_data_to_dict(data):
    final_dict ={}
    if len(data.shape) == 3:
        time_pts, body_parts, axes = data.shape
        num_cams = 1
    elif len(data.shape) == 4:
        num_cams, time_pts, body_parts, axes = data.shape
        cams_dict = {}
    if body_parts != len(tracked_joints):
        raise Exception("Check tracked joints definition")

    for cam in range(num_cams):
        exp_dict = {}
        if num_cams > 1:
            coords = data[cam]
        else:
            coords = data
        for i, name in enumerate(tracked_joints):
            if 'Antenna' in name: 
                body_part = 'Antennae'
                landmark = name
            elif 'Stripe' in name:
                body_part = 'Stripes'
                landmark = name
            else:
                body_part = name[0:2] + '_leg'
                landmark = name[2:]

            if body_part not in exp_dict.keys():
                exp_dict[body_part] = {}

            exp_dict[body_part][landmark] = coords[:,i,:]

        if num_cams>1:
            cams_dict[cam] = exp_dict

    if num_cams>1:
        final_dict = cams_dict
    else:
        final_dict = exp_dict
        
    return final_dict


def convert_to_df3d_output_format(aligned, fixed=True):
    if fixed:
        key = "fixed_pos_aligned"
    else:
        key = "raw_pos_aligned"

    n_timepoints = len(aligned["LF_leg"]["Femur"]["raw_pos_aligned"])

    points3D = []

    def upsample_fixed_coxa(points, n_timepoints):
        if points.ndim == 2:
            return points
        points = points[np.newaxis]
        points = np.repeat(points, n_timepoints, axis=0)
        return points

    points3D.append(upsample_fixed_coxa(aligned["LF_leg"]["Coxa"][key], n_timepoints))
    points3D.append(aligned["LF_leg"]["Femur"]["raw_pos_aligned"])
    points3D.append(aligned["LF_leg"]["Tibia"]["raw_pos_aligned"])
    points3D.append(aligned["LF_leg"]["Tarsus"]["raw_pos_aligned"])
    points3D.append(aligned["LF_leg"]["Claw"]["raw_pos_aligned"])

    points3D.append(upsample_fixed_coxa(aligned["LM_leg"]["Coxa"][key], n_timepoints))
    points3D.append(aligned["LM_leg"]["Femur"]["raw_pos_aligned"])
    points3D.append(aligned["LM_leg"]["Tibia"]["raw_pos_aligned"])
    points3D.append(aligned["LM_leg"]["Tarsus"]["raw_pos_aligned"])
    points3D.append(aligned["LM_leg"]["Claw"]["raw_pos_aligned"])

    points3D.append(upsample_fixed_coxa(aligned["LH_leg"]["Coxa"][key], n_timepoints))
    points3D.append(aligned["LH_leg"]["Femur"]["raw_pos_aligned"])
    points3D.append(aligned["LH_leg"]["Tibia"]["raw_pos_aligned"])
    points3D.append(aligned["LH_leg"]["Tarsus"]["raw_pos_aligned"])
    points3D.append(aligned["LH_leg"]["Claw"]["raw_pos_aligned"])

    # Left antenna
    points3D.append(np.zeros_like(aligned["LH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)
    # Left stripe 1
    points3D.append(np.zeros_like(aligned["LH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)
    # Left stripe 2
    points3D.append(np.zeros_like(aligned["LH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)
    # Left stripe 3
    points3D.append(np.zeros_like(aligned["LH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)

    points3D.append(upsample_fixed_coxa(aligned["RF_leg"]["Coxa"][key], n_timepoints))
    points3D.append(aligned["RF_leg"]["Femur"]["raw_pos_aligned"])
    points3D.append(aligned["RF_leg"]["Tibia"]["raw_pos_aligned"])
    points3D.append(aligned["RF_leg"]["Tarsus"]["raw_pos_aligned"])
    points3D.append(aligned["RF_leg"]["Claw"]["raw_pos_aligned"])

    points3D.append(upsample_fixed_coxa(aligned["RM_leg"]["Coxa"][key], n_timepoints))
    points3D.append(aligned["RM_leg"]["Femur"]["raw_pos_aligned"])
    points3D.append(aligned["RM_leg"]["Tibia"]["raw_pos_aligned"])
    points3D.append(aligned["RM_leg"]["Tarsus"]["raw_pos_aligned"])
    points3D.append(aligned["RM_leg"]["Claw"]["raw_pos_aligned"])

    points3D.append(upsample_fixed_coxa(aligned["RH_leg"]["Coxa"][key], n_timepoints))
    points3D.append(aligned["RH_leg"]["Femur"]["raw_pos_aligned"])
    points3D.append(aligned["RH_leg"]["Tibia"]["raw_pos_aligned"])
    points3D.append(aligned["RH_leg"]["Tarsus"]["raw_pos_aligned"])
    points3D.append(aligned["RH_leg"]["Claw"]["raw_pos_aligned"])

    # Right antenna
    points3D.append(np.zeros_like(aligned["RH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)
    # Right stripe 1
    points3D.append(np.zeros_like(aligned["RH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)
    # Right stripe 2
    points3D.append(np.zeros_like(aligned["RH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)
    # Right stripe 3
    points3D.append(np.zeros_like(aligned["RH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan)

    points3D = np.array(points3D)
    points3D = np.swapaxes(points3D, 0, 1)
    return points3D


def angles_as_list(angles):
    angle_list = []

    angle_list.append(angles["LF_leg"]["yaw"])
    angle_list.append(angles["LF_leg"]["pitch"])
    angle_list.append(angles["LF_leg"]["roll"])
    angle_list.append(angles["LF_leg"]["th_fe"])
    angle_list.append(angles["LF_leg"]["th_ti"])
    angle_list.append(angles["LF_leg"]["th_ta"])
    
    angle_list.append(angles["LM_leg"]["yaw"])
    angle_list.append(angles["LM_leg"]["pitch"])
    angle_list.append(angles["LM_leg"]["roll"])
    angle_list.append(angles["LM_leg"]["th_fe"])
    angle_list.append(angles["LM_leg"]["th_ti"])
    angle_list.append(angles["LM_leg"]["th_ta"])
    
    angle_list.append(angles["LH_leg"]["yaw"])
    angle_list.append(angles["LH_leg"]["pitch"])
    angle_list.append(angles["LH_leg"]["roll"])
    angle_list.append(angles["LH_leg"]["th_fe"])
    angle_list.append(angles["LH_leg"]["th_ti"])
    angle_list.append(angles["LH_leg"]["th_ta"])
    
    angle_list.append(angles["RF_leg"]["yaw"])
    angle_list.append(angles["RF_leg"]["pitch"])
    angle_list.append(angles["RF_leg"]["roll"])
    angle_list.append(angles["RF_leg"]["th_fe"])
    angle_list.append(angles["RF_leg"]["th_ti"])
    angle_list.append(angles["RF_leg"]["th_ta"])
    
    angle_list.append(angles["RM_leg"]["yaw"])
    angle_list.append(angles["RM_leg"]["pitch"])
    angle_list.append(angles["RM_leg"]["roll"])
    angle_list.append(angles["RM_leg"]["th_fe"])
    angle_list.append(angles["RM_leg"]["th_ti"])
    angle_list.append(angles["RM_leg"]["th_ta"])
    
    angle_list.append(angles["RH_leg"]["yaw"])
    angle_list.append(angles["RH_leg"]["pitch"])
    angle_list.append(angles["RH_leg"]["roll"])
    angle_list.append(angles["RH_leg"]["th_fe"])
    angle_list.append(angles["RH_leg"]["th_ti"])
    angle_list.append(angles["RH_leg"]["th_ta"])

    angle_list = np.array(angle_list)
    angle_list = np.swapaxes(angle_list, 0, 1)
    
    return angle_list
