import itertools

import numpy as np
import os
from .utils.utils_alignment import align_3d
from .utils.utils_angles import calculate_angles
#from .utils.utils_plots import *
import deepfly.signal_util

df3d_skeleton = ['LFCoxa',
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

prism_skeleton = ['LFCoxa',
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
                  'RHClaw',
                  'LAntenna',
                  'RAntenna',
                  'LEye',
                  'REye',
                  'LHaltere',
                  'RHaltere',
                  'LWing',
                  'RWing',
                  'Proboscis',
                  'Neck',
                  'Genitalia',
                  'Scutellum',
                  'A1A2',
                  'A3',
                  'A4',
                  'A5',
                  'A6']

template_neuromechfly = {'LFCoxa':[0.345, -0.216, 0.327],
                         'LFFemur':[0.345, -0.216, -0.025],
                         'LFTibia':[0.345, -0.216, -0.731],
                         'LFTarsus':[0.345, -0.216, -1.249],
                         'LFClaw':[0.345, -0.216, -1.912],
                         'RFCoxa':[0.345, 0.216, 0.327],
                         'RFFemur':[0.345, 0.216, -0.025],
                         'RFTibia':[0.345, 0.216, -0.731],
                         'RFTarsus':[0.345, 0.216, -1.249],
                         'RFClaw':[0.345, 0.216, -1.912],
                         'LMCoxa':[0, -0.125, 0],
                         'LMFemur':[0, -0.125, -0.182],
                         'LMTibia':[0, -0.125, -0.965],
                         'LMTarsus':[0, -0.125, -1.633],
                         'LMClaw':[0, -0.125, -2.328],
                         'RMCoxa':[0, 0.125, 0],
                         'RMFemur':[0, 0.125, -0.182],
                         'RMTibia':[0, 0.125, -0.965],
                         'RMTarsus':[0, 0.125, -1.633],
                         'RMClaw':[0, 0.125, -2.328],                      
                         'LHCoxa':[-0.215, -0.087, -0.073],
                         'LHFemur':[-0.215, -0.087, -0.272],
                         'LHTibia':[-0.215, -0.087, -1.108],
                         'LHTarsus':[-0.215, -0.087, -1.793],
                         'LHClaw':[-0.215, -0.087, -2.588],                      
                         'RHCoxa':[-0.215, 0.087, -0.073],
                         'RHFemur':[-0.215, 0.087, -0.272],
                         'RHTibia':[-0.215, 0.087, -1.108],
                         'RHTarsus':[-0.215, 0.087, -1.793],
                         'RHClaw':[-0.215, 0.087, -2.588]}

class df3dPostProcess:
    def __init__(self, exp_dir, multiple = False, file_name = '', skeleton='df3d', calculate_3d=False):
        self.exp_dir = exp_dir
        self.raw_data_3d = np.array([])
        self.raw_data_2d = np.array([])
        self.skeleton = skeleton
        self.template = template_neuromechfly
        self.raw_data_cams = {}
        self.load_data(exp_dir, calculate_3d, skeleton, multiple, file_name)
        self.data_3d_dict = load_data_to_dict(self.raw_data_3d, skeleton)
        self.data_2d_dict = load_data_to_dict(self.raw_data_2d, skeleton)
        self.aligned_model = {}

    def align_to_template(self,scale=True):
        self.aligned_model = align_3d(self.data_3d_dict,self.skeleton,self.template,scale)

        return self.aligned_model

    #def align_3d_data(self, rescale = True):
    #    self.aligned_model = align_data(self.data_3d_dict,self.skeleton)
    #    if rescale:
    #        self.aligned_model = rescale_using_2d_data(self.aligned_model, self.data_2d_dict, self.raw_data_cams, self.exp_dir)       
    #    return self.aligned_model

    def calculate_leg_angles(self, begin = 0, end = 0, get_roll_tr = True):
        leg_angles = calculate_angles(self.aligned_model, begin, end, get_roll_tr)
        return leg_angles

    
    def load_data(self, exp, calculate_3d, skeleton, multiple, file_name):
        if multiple:
            currentDirectory = os.getcwd()
            dataFile = os.path.join(currentDirectory,file_name)
            data = np.load(dataFile,allow_pickle=True)
            self.raw_data_3d  = data[exp]
        else:
            data = np.load(exp,allow_pickle=True)
            if skeleton == 'prism':
                data={'points3d':data}
            for key, vals in data.items():
                if not isinstance(key,str):
                    self.raw_data_cams[key] = vals
                elif key == 'points3d':
                    if calculate_3d:
                        self.raw_data_3d = triangulate_2d(data, exp)
                    else:
                        self.raw_data_3d = vals
                elif key == 'points2d':
                    self.raw_data_2d = vals

def triangulate_2d(data, exp_dir):
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

    camNet = correct_outliers(camNet)

    return camNet.points3d


def correct_outliers(camNet):
    from deepfly.cv_util import triangulate_linear

    #outlier_image_ids = np.where(np.abs(camNet.points3d) > 5)[0]
    #outlier_joint_ids = np.where(np.abs(camNet.points3d) > 5)[1]
    #print(outlier_image_ids)
    #print(outlier_joint_ids)

    start_indices = np.array([0, 1, 2, 3,
                              5, 6, 7, 8,
                              10, 11, 12, 13,
                              19, 20, 21, 22,
                              24, 25, 26, 27,
                              29, 30, 31, 32,],
                             dtype=np.int)
    stop_indices = start_indices + 1
    #pairs = np.vstack((start_indices, stop_indices))
    #lengths = np.sum((camNet.points3d[:, start_indices, :] - camNet.points3d[:, stop_indices, :]) ** 2, axis=2)
    lengths = np.linalg.norm(camNet.points3d[:, start_indices, :] - camNet.points3d[:, stop_indices, :], axis=2)
    median_lengths = np.median(lengths, axis=0)
    #length_outliers = (lengths > np.quantile(lengths, 0.99, axis=0)) | (lengths < np.quantile(lengths, 0.01, axis=0))
    length_outliers = (lengths > median_lengths * 1.4) | (lengths < median_lengths * 0.4)
    #print(np.quantile(lengths, 0.99, axis=0))
    #print(np.max(lengths, axis=0))
    #print(np.quantile(lengths, 0.01, axis=0))
    #print(np.min(lengths, axis=0))
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
    #print(outlier_image_ids)
    #print(outlier_joint_ids)
    #exit()
    

    def _triangluate_specific_cameras(camNet, cam_id_list,img_id, j_id):
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
        # Select cameras based on which side the joint is on
        all_cam_ids = [0, 1, 2] if joint_id < 19 else [4, 5, 6]
        for subset_cam_ids in itertools.combinations(all_cam_ids, 2):
            points3d_using_2_cams = _triangluate_specific_cameras(camNet, subset_cam_ids, img_id, joint_id)

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

        #reprojection_error_function = lambda cam_id: camNet.cam_list[cam_id].project(camNet.points3d[img_id, joint_id]) - camNet.cam_list[cam_id].points2d[img_id, joint_id]
        #reprojection_error_3_cams = np.mean([reprojection_error_function(cam_id) for cam_id in all_cam_ids])

        #if np.min(reprojection_errors) < reprojection_error_3_cams:
        # Replace 3D points with best estimation from 2 cameras only
        #best_cam_tuple_index = np.argmin(reprojection_errors)
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
        #else:
        #    pass
            #print("not correcting", img_id, joint_id, new_diff, old_diff)
        #if img_id == 18957:
        #    print(segment_length_diff)
        #    print(new_diff)
        #    print(old_diff)
        #    exit()
    #print(np.histogram(np.max(camNet.points3d[:, :, 0], axis=1)))
    #exit()
    #center_frame_index = np.argmax(np.max(camNet.points3d[:, :, 0], axis=1))
    #center_frame_index = 12801
    #center_frame_index = 18957
    #print(center_frame_index)
    #camNet.points3d = camNet.points3d[max(0, center_frame_index - 50):min(camNet.points3d.shape[0], center_frame_index + 50)]
    #exit()
    #print(camNet.points3d.shape)
    camNet.points3d = deepfly.signal_util.filter_batch(camNet.points3d, freq=100)
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
                
            '''    
            if 'Antenna' in name: 
                body_part = 'Antennae'
                landmark = name
            elif 'Stripe' in name:
                body_part = 'Stripes'
                landmark = name
            else:
                body_part = name[0:2] + '_leg'
                landmark = name[2:]
            '''
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






