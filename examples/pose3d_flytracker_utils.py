import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import pickle
from scipy.interpolate import interp1d # Added by Ari for interpolation

def save_file(out_fname, data):
    with open(out_fname, 'wb') as f:
        pickle.dump(data, f)
        

def fill_missing_joints(df, plot = False):
    """
    Goes through df and extrapolates to give values for NaN joints
    
    """
    
    for key in df.keys(): # Iterate through columns
        if key == 'Frame_idx':
            continue
            
        nan_ids = np.where(df[key].isna())[0]
        
        if list(nan_ids)==[]:
            continue # No need for interpolation if there are no nans
            
        # Interpolation  
        column = df[key]
        frames = df['Frame_idx']
        cut_column = column.drop(np.where(df[key].isna())[0]).reset_index(drop=True).to_numpy() # Joint column but without the nans
        cut_frames = frames.drop(np.where(df[key].isna())[0]).reset_index(drop=True).to_numpy() # Frames without nans for that joint

        # Interpolate
        pos_func = interp1d(cut_frames,cut_column, fill_value = 'extrapolate')
        df[key]= pos_func(frames)

        if plot:
            plt.figure()
            plt.plot(cut_frames, cut_column, label='no nan data')
            plt.plot(frames, pos_func(frames),'go',label='extrapolation')
            plt.legend()
            plt.show()
        
    return df


def df_to_array(df, frames = False, save_array = False, save_as = None):
    
    """Takes the df of joints from flytracker and 
    converts it to a numpy array with shape (N_frames, total_joints, 3)
    to be compatible with the df3d joint angle calculator"""
    
    leg_keys = ['RF_leg', 'RM_leg', 'RH_leg', 'LF_leg', 'LM_leg', 'LH_leg'] # inverse kinematics keys
    joints = ['ThC', 'CTr','FTi','TiTa','Claw']
    num_frames = df.shape[0] 

    # Convert to appropriate format

    pose_list = [] # Initialise an empty dict for the cartesian (ijk) coords

    # MULTIPLE FRAMES

    for df_id in range(num_frames): # 2. Loop through frames

        frame_list = []
        
        for leg_key in leg_keys:# 1. Loop through legs

            leg = leg_key[:2] # RF, RM, etc.

            for joint in joints: # 3. Loop through joints

                joint_sf = leg + '-' + joint + '_' # Joint string format
                joint_row = []

                for coord in ['x','y','z']:

                    column = joint_sf+coord
                    joint_row.append(df[column][df_id])

                frame_list.append(joint_row) 

        pose_list.append(frame_list)

    pose_array = np.array(pose_list)
    
    frame_inds = df['Frame_idx'].to_numpy()
    
    if save_array:
            save_file(save_as, pose_array)
    
    if frames:
        return pose_array, frame_inds
    
    else:
        return pose_array
    
    
def sparse_to_dense_angles(angles, frame_inds, raw_fps = 25, dense_dt = 0.01, save_dense_angles = False, save_as = None):
    
    """ Performs extrapolation to match NMF's 1000Hz standards
        save_path should end in .pkl
    """
    
    t = frame_inds/raw_fps
    dense_t = np.arange(t[0], t[-1], step = dense_dt)
    
    leg_keys = ['RF_leg', 'RM_leg', 'RH_leg', 'LF_leg', 'LM_leg', 'LH_leg']
    angle_keys = ['ThC_yaw', 'ThC_pitch', 'ThC_roll', 'CTr_pitch', 'FTi_pitch', 'CTr_roll', 'TiTa_pitch']
    
    angles_dense = {}

    for leg in leg_keys:

        angles_dense[f"{leg}"]={}

        for angle_name in angle_keys:

            joint_angles = angles[f"{leg}"][f"{angle_name}"] # angles for all frames for one specific joint

            # For vel we have to compare two subsequent frames
            pos_func = interp1d(t, joint_angles, fill_value = "extrapolate")
            angles_dense[f"{leg}"][f"{angle_name}"] = pos_func(dense_t).tolist()
            
    if save_dense_angles:
        
        save_file(save_as, angles_dense)
    
    return angles_dense



