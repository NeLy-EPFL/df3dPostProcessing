import numpy as np
import os
from .utils.utils_alignment import align_data, rescale_using_2d_data, fixed_lengths_and_base_point
from .utils.utils_angles import calculate_angles
#from .utils.utils_plots import *

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
    column_names = []

    def upsample_fixed_coxa(points, n_timepoints):
        if points.ndim == 2:
            return points
        points = points[np.newaxis]
        points = np.repeat(points, n_timepoints, axis=0)
        return points

    missing_value_placeholder = np.zeros_like(aligned["LH_leg"]["Claw"]["raw_pos_aligned"]) * np.nan

    for leg in ["LF_leg", "LM_leg", "LH_leg", "L_antenna", "L_stripes", "RF_leg", "RM_leg", "RH_leg", "R_antenna", "R_stripes"]:
        if "antenna" in leg:
            points3D.append(missing_value_placeholder)
            column_names.append(leg)
        elif "stripes" in leg:
            for stripe_number in range(3):
                points3D.append(missing_value_placeholder)
                column_name = " ".join([leg, f"{stripe_number}"])
                column_names.append(column_name)
        else:
            for joint in ["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]:
                column_name = " ".join([leg, joint])
                column_names.append(column_name)
                if joint == "Coxa":
                    points3D.append(upsample_fixed_coxa(aligned[leg]["Coxa"][key], n_timepoints))
                else:
                    points3D.append(aligned[leg][joint]["raw_pos_aligned"])

    points3D = np.array(points3D)
    points3D = np.swapaxes(points3D, 0, 1)
    return points3D, column_names


def angles_as_list(angles):
    angle_list = []
    column_names = []

    for leg in ["LF_leg", "LM_leg", "LH_leg", "RF_leg", "RM_leg", "RH_leg"]:
        for angle in ["yaw", "pitch", "roll", "th_fe", "th_ti", "roll_tr", "th_ta"]:
            angle_list.append(angles[leg][angle])
            column_names.append(" ".join([leg, angle]))

    angle_list = np.array(angle_list)
    angle_list = np.swapaxes(angle_list, 0, 1)
    
    return angle_list, column_names


def angle_list_to_dict(angles):
    angle_dict = {}
    current_index = 0
    for leg in ["LF_leg", "LM_leg", "LH_leg", "RF_leg", "RM_leg", "RH_leg"]:
        angle_dict[leg] = {}
        for angle in ["yaw", "pitch", "roll", "th_fe", "th_ti", "roll_tr", "th_ta"]:
            angle_dict[leg][angle] = angles[:, current_index]
            current_index += 1
    if current_index != angles.shape[1]:
        raise ValueError("Second dimensions of angles matrix does not match the number of known angles.")
    return angle_dict


def aligned_points_to_dict(data):
    data_dict = load_data_to_dict(data)
    data_dict = fixed_lengths_and_base_point(data_dict)
    aligned_dict = {}
    for leg, leg_data in data_dict.items():
        if "leg" not in leg:
            continue
        aligned_dict[leg] = {}
        for segment, segment_data in leg_data.items():
            aligned_dict[leg][segment] = {}
            try:
                aligned_dict[leg][segment]["mean_length"] = segment_data["mean_length"]
            except KeyError:
                pass
            segment_data = segment_data["raw_pos"]
            if segment == "Coxa" and np.unique(segment_data, axis=0).shape[0] == 1:
                aligned_dict[leg][segment]["fixed_pos_aligned"] = segment_data[0]
            else:
                aligned_dict[leg][segment]["raw_pos_aligned"] = segment_data
    return aligned_dict
