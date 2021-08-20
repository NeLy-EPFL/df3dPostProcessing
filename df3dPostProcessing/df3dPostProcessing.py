import numpy as np
import os
import pickle
import scipy.signal
import itertools
from df3dPostProcessing.utils.utils_alignment import align_3d
from df3dPostProcessing.utils.utils_angles import calculate_angles
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

prism_skeleton_AG = ['LFCoxa',
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

template_nmf = {'RFCoxa':[0.35, -0.27, 0.400],
                'RFFemur':[0.35, -0.27, -0.025],
                'RFTibia':[0.35, -0.27, -0.731],
                'RFTarsus':[0.35, -0.27, -1.249],
                'RFClaw':[0.35, -0.27, -1.912],
                'LFCoxa':[0.35, 0.27, 0.400],
                'LFFemur':[0.35, 0.27, -0.025],
                'LFTibia':[0.35, 0.27, -0.731],
                'LFTarsus':[0.35, 0.27, -1.249],
                'LFClaw':[0.35, 0.27, -1.912],
                'RMCoxa':[0, -0.125, 0],
                'RMFemur':[0, -0.125, -0.182],
                'RMTibia':[0, -0.125, -0.965],
                'RMTarsus':[0, -0.125, -1.633],
                'RMClaw':[0, -0.125, -2.328],
                'LMCoxa':[0, 0.125, 0],
                'LMFemur':[0, 0.125, -0.182],
                'LMTibia':[0, 0.125, -0.965],
                'LMTarsus':[0, 0.125, -1.633],
                'LMClaw':[0, 0.125, -2.328],                      
                'RHCoxa':[-0.215, -0.087, -0.073],
                'RHFemur':[-0.215, -0.087, -0.272],
                'RHTibia':[-0.215, -0.087, -1.108],
                'RHTarsus':[-0.215, -0.087, -1.793],
                'RHClaw':[-0.215, -0.087, -2.588],                      
                'LHCoxa':[-0.215, 0.087, -0.073],
                'LHFemur':[-0.215, 0.087, -0.272],
                'LHTibia':[-0.215, 0.087, -1.108],
                'LHTarsus':[-0.215, 0.087, -1.793],
                'LHClaw':[-0.215, 0.087, -2.588],
                'RAntenna':[0.25, -0.068, 0.67],
                'LAntenna':[0.25, 0.068, 0.67]}

zero_pose_nmf = {'RF_leg':{'ThC_yaw':0,
                           'ThC_pitch':0,
                           'ThC_roll':0,
                           'CTr_roll':0,
                           'CTr_pitch':-np.pi,
                           'FTi_pitch':np.pi,
                           'TiTa_pitch':-np.pi},
                 'RM_leg':{'ThC_yaw':0,
                           'ThC_pitch':0,
                           'ThC_roll':-np.pi/2,
                           'CTr_roll':0,
                           'CTr_pitch':-np.pi,
                           'FTi_pitch':np.pi,
                           'TiTa_pitch':-np.pi},
                 'RH_leg':{'ThC_yaw':0,
                           'ThC_pitch':0,
                           'ThC_roll':-np.pi,
                           'CTr_roll':0,
                           'CTr_pitch':-np.pi,
                           'FTi_pitch':np.pi,
                           'TiTa_pitch':-np.pi},
                 'LF_leg':{'ThC_yaw':0,
                           'ThC_pitch':0,
                           'ThC_roll':0,
                           'CTr_roll':0,
                           'CTr_pitch':-np.pi,
                           'FTi_pitch':np.pi,
                           'TiTa_pitch':-np.pi},
                 'LM_leg':{'ThC_yaw':0,
                           'ThC_pitch':0,
                           'ThC_roll':np.pi/2,
                           'CTr_roll':0,
                           'CTr_pitch':-np.pi,
                           'FTi_pitch':np.pi,
                           'TiTa_pitch':-np.pi},
                 'LH_leg':{'ThC_yaw':0,
                           'ThC_pitch':0,
                           'ThC_roll':np.pi,
                           'CTr_roll':0,
                           'CTr_pitch':-np.pi,
                           'FTi_pitch':np.pi,
                           'TiTa_pitch':-np.pi}}

class df3dPostProcess:
    def __init__(self, exp_dir, multiple = False, file_name = '', skeleton='df3d', calculate_3d=False, correct_outliers=False):
        self.exp_dir = exp_dir
        self.raw_data_3d = np.array([])
        self.raw_data_2d = np.array([])
        self.skeleton = skeleton
        self.template = template_nmf
        self.zero_pose = zero_pose_nmf
        self.raw_data_cams = {}
        self.load_data(exp_dir, calculate_3d, skeleton, multiple, file_name, correct_outliers)
        self.data_3d_dict = load_data_to_dict(self.raw_data_3d, skeleton)
        self.data_2d_dict = load_data_to_dict(self.raw_data_2d, skeleton)
        self.aligned_model = {}

    def align_to_template(self,scale=True,all_body=False,interpolate=False,smoothing=True,original_time_step= 0.01,new_time_step=0.001,window_length=29):
        self.aligned_model = align_3d(self.data_3d_dict,self.skeleton,self.template,scale,all_body,interpolate, smoothing, original_time_step, new_time_step, window_length)

        return self.aligned_model

    def calculate_leg_angles(self, begin = 0, end = 0, get_CTr_roll = True, save_angles=False, use_zero_pose=True, zero_pose=None):
        if not zero_pose:
            zero_pose = self.zero_pose
                
        leg_angles = calculate_angles(self.aligned_model, begin, end, get_CTr_roll, zero_pose)

        if use_zero_pose:
            for leg, angles in leg_angles.items():
                for angle, vals in angles.items():
                    leg_angles[leg][angle] = list(np.array(vals) + zero_pose[leg][angle])
                
        if save_angles:
            path = self.exp_dir.replace('pose_result','joint_angles')
            with open(path, 'wb') as f:
                pickle.dump(leg_angles, f)
        return leg_angles

    def calculate_velocity(self, data, window=11, order=3, time_step=0.001, save_velocities=False):
        velocities = {}
        for leg, angles in data.items():
            velocities[leg]={}
            for th, data in angles.items():
                vel = scipy.signal.savgol_filter(data, window, order, deriv=1, delta=time_step, mode='nearest')
                velocities[leg][th]=vel

        if save_velocities:
            path = self.exp_dir.replace('pose_result','joint_velocities')
            with open(path, 'wb') as f:
                pickle.dump(velocities, f)
            
        return velocities

    
    def load_data(self, exp, calculate_3d, skeleton, multiple, file_name, correct_outliers=False):
        if multiple:
            currentDirectory = os.getcwd()
            dataFile = os.path.join(currentDirectory,file_name)
            data = np.load(dataFile,allow_pickle=True)
            self.raw_data_3d  = data[exp]
        else:
            data = np.load(exp,allow_pickle=True)
            if skeleton == 'prism':
                data={'points3d':data}
            #from IPython import embed; embed()
            if calculate_3d:
                self.raw_data_3d = triangulate_2d(data, exp, correct_outliers)
                
            for key, vals in data.items():
                if not isinstance(key,str):
                    self.raw_data_cams[key] = vals
                elif key == 'points3d':
                    if not calculate_3d:
                        self.raw_data_3d = vals['points3d']
                elif key == 'points2d':
                    self.raw_data_2d = vals

def triangulate_2d(data, exp_dir, correct_outliers=False):
    img_folder = exp_dir[:exp_dir.find('df3d')]
    out_folder = exp_dir[:exp_dir.find('pose_result')]
    num_images = data['points3d'].shape[0]

    from deepfly.CameraNetwork import CameraNetwork

    camNet = CameraNetwork(image_folder=img_folder, output_folder=out_folder, num_images=num_images)

    for cam_id in range(7): 
        camNet.cam_list[cam_id].set_intrinsic(data[cam_id]["intr"]) 
        camNet.cam_list[cam_id].set_R(data[cam_id]["R"]) 
        camNet.cam_list[cam_id].set_tvec(data[cam_id]["tvec"]) 
    camNet.triangulate() 

    if correct_outliers:
        camNet = correct_outliers_legs(camNet)

    return camNet.points3d


def correct_outliers_legs(camNet):
    """ Corrects outlier keypoint estimations. """
    from deepfly.signal_util import filter_batch
    from deepfly.cv_util import triangulate_linear

    start_indices = np.array([0, 1, 2, 3,
                                5, 6, 7, 8,
                                10, 11, 12, 13,
                                19, 20, 21, 22,
                                24, 25, 26, 27,
                                29, 30, 31, 32,],
                                dtype=np.int)
    stop_indices = start_indices + 1
    #pairs = np.vstack((start_indices, stop_indices))
    lengths = np.linalg.norm(camNet.points3d[:, start_indices, :] - camNet.points3d[:, stop_indices, :], axis=2)
    median_lengths = np.median(lengths, axis=0)
    #length_outliers = (lengths > np.quantile(lengths, 0.99, axis=0)) | (lengths < np.quantile(lengths, 0.01, axis=0))
    length_outliers = (lengths > median_lengths * 1.4) | (lengths < median_lengths * 0.4)

    outlier_mask = np.zeros(camNet.points3d.shape, dtype=np.bool)
    for i, mask_offset in enumerate([0, 5, 10, 19, 24, 29]):
        claw_outliers = np.where(length_outliers[:, i * 4 + 3] & ~length_outliers[:, i * 4 + 2])[0]
        tarsus_outliers = np.where(length_outliers[:, i * 4 + 3] & length_outliers[:, i * 4 + 2])[0]
        tibia_outliers = np.where(length_outliers[:, i * 4 + 2] & length_outliers[:, i * 4 + 1])[0]
        femur_outliers = np.where(length_outliers[:, i * 4 + 1] & length_outliers[:, i * 4 + 0])[0]

        outlier_mask[femur_outliers, 1 + mask_offset] = True
        outlier_mask[tibia_outliers, 2 + mask_offset] = True
        outlier_mask[tarsus_outliers, 3 + mask_offset] = True
        outlier_mask[claw_outliers, 4 + mask_offset] = True

    outlier_image_ids = np.where(outlier_mask)[0]
    outlier_joint_ids = np.where(outlier_mask)[1]
    # print(outlier_image_ids)
    # print(outlier_joint_ids)

    # Camera order starts from right hind side to the left
    cam_list = camNet.cid2cidread

    def _triangulate_specific_cameras(camNet, cam_id_list,img_id, j_id):
        cam_list_iter = list()
        points2d_iter = list()
        for cam in [camNet.cam_list[cam_idx] for cam_idx in cam_id_list]:
            cam_list_iter.append(cam)
            points2d_iter.append(cam[img_id, j_id, :])
        return triangulate_linear(cam_list_iter, points2d_iter)

    for img_id, joint_id in zip(outlier_image_ids, outlier_joint_ids):
        reprojection_errors = list()
        segment_length_diff = list()
        points_using_2_cams = list()
        # Select cameras based on which side the joint is on, joint < 19 is the right side
        all_cam_ids = cam_list[:3] if joint_id < 19 else cam_list[-3:]

        for subset_cam_ids in itertools.combinations(all_cam_ids, 2):
            points3d_using_2_cams = _triangulate_specific_cameras(camNet, subset_cam_ids, img_id, joint_id)

            new_diff = 0
            median_index = np.where(stop_indices == joint_id)[0]
            if len(median_index) > 0:
                new_diff += np.linalg.norm(points3d_using_2_cams - camNet.points3d[img_id, joint_id - 1]) - median_lengths[median_index]
            median_index = np.where(start_indices == joint_id)[0]
            if len(median_index) > 0:
                new_diff += np.linalg.norm(points3d_using_2_cams - camNet.points3d[img_id, joint_id + 1]) - median_lengths[median_index]
            segment_length_diff.append(new_diff)

            reprojection_error_function = lambda cam_id: camNet.cam_list[cam_id].project(points3d_using_2_cams) - camNet.cam_list[cam_id].points2d[img_id, joint_id]
            reprojection_error = np.mean([reprojection_error_function(cam_id) for cam_id in subset_cam_ids])
            reprojection_errors.append(reprojection_error)
            points_using_2_cams.append(points3d_using_2_cams)

        # Replace 3D points with best estimation from 2 cameras only
        best_cam_tuple_index = np.argmin(segment_length_diff)

        old_diff = 0
        new_diff = 0
        median_index = np.where(stop_indices == joint_id)[0]
        if len(median_index) > 0:
            #print(pairs[:, median_index], joint_id, joint_id - 1)
            old_diff += np.linalg.norm(camNet.points3d[img_id, joint_id] - camNet.points3d[img_id, joint_id - 1]) - median_lengths[median_index]
            new_diff += np.linalg.norm(points_using_2_cams[best_cam_tuple_index] - camNet.points3d[img_id, joint_id - 1]) - median_lengths[median_index]
        median_index = np.where(start_indices == joint_id)[0]
        if len(median_index) > 0:
            #print(pairs[:, median_index], joint_id, joint_id + 1)
            old_diff += np.linalg.norm(camNet.points3d[img_id, joint_id] - camNet.points3d[img_id, joint_id + 1]) - median_lengths[median_index]
            new_diff += np.linalg.norm(points_using_2_cams[best_cam_tuple_index] - camNet.points3d[img_id, joint_id + 1]) - median_lengths[median_index]

        if new_diff < old_diff:
            #print("correcting", img_id, joint_id)
            camNet.points3d[img_id, joint_id] = points_using_2_cams[best_cam_tuple_index]

    camNet.points3d = filter_batch(camNet.points3d, freq=100)
    return camNet


def load_data_to_dict(data, skeleton):
    final_dict ={}
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

            exp_dict[body_part][landmark] = coords[:,i,:]

        if num_cams>1:
            cams_dict[cam] = exp_dict

    if num_cams>1:
        final_dict = cams_dict
    else:
        final_dict = exp_dict
        
    return final_dict






