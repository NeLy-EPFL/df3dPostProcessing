import numpy as np
import os
from .utils.utils_alignment import align_data, rescale_using_2d_data
from .utils.utils_angles import calculate_angles
#from .utils.utils_plots import *
from scipy.interpolate import pchip_interpolate

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
                  'RHClaw']

class df3dPostProcess:
    def __init__(self, results_dir, multiple = False, file_name = '', skeleton='df3d'):
        self.res_dir = results_dir
        self.raw_data_3d = np.array([])
        self.raw_data_2d = np.array([])
        self.skeleton = skeleton
        self.raw_data_cams = {}
        self.load_data(results_dir, multiple, file_name)
        self.data_3d_dict = load_data_to_dict(self.raw_data_3d, skeleton)
        self.data_2d_dict = load_data_to_dict(self.raw_data_2d, skeleton)
        self.aligned_model = {}


    def interpolate_smooth(self,signal,sample_rate=0.01,new_sample_rate=0.001,begin=0,smoothing=True, window_length=29):
        """ This function interpolates the signal based on PCHIP and smoothes it using Hamming window. """
        end = begin + len(signal)*sample_rate
        x = np.arange(begin,end,sample_rate)
        x_new = np.arange(begin,end,new_sample_rate)
        interpolated_signal = pchip_interpolate(x,signal,x_new)
        if smoothing:
            hamming_window = np.hamming(window_length)
            y = np.convolve(hamming_window/hamming_window.sum(), interpolated_signal, mode='valid')
            return y

        return interpolated_signal

    def align_3d_data(self, rescale = True, interpolation=True, smoothing=True, begin=0,sample_rate = 0.01, new_sample_rate=0.001):
        """ If interpolate option is choosed, this function interpolates the data given the x points and sample rates."""
        aligned_model = align_data(self.data_3d_dict,self.skeleton)
        if rescale:
            aligned_model = rescale_using_2d_data(aligned_model, self.data_2d_dict, self.raw_data_cams, self.res_dir)
        if interpolation:
            aligned_dic = {key: {key2: {key3: list() for key3 in aligned_model[key][key2].keys()} for key2 in aligned_model[key].keys()} for key in aligned_model.keys()}
            for key in aligned_dic.keys():
                for key2 in aligned_dic[key].keys():
                    for key3 in aligned_dic[key][key2].keys():
                        if key3=='raw_pos_aligned':
                            intp_x = self.interpolate_smooth(aligned_model[key][key2][key3][:,0],0.01,0.001,0,smoothing, window_length=29)
                            intp_y = self.interpolate_smooth(aligned_model[key][key2][key3][:,1],0.01,0.001,0,smoothing, window_length=29)
                            intp_z = self.interpolate_smooth(aligned_model[key][key2][key3][:,2],0.01,0.001,0,smoothing, window_length=29)

                            aligned_array = np.array([intp_x, intp_y, intp_z])
                            aligned_dic[key][key2][key3] = aligned_array.T
                        else:
                            aligned_dic[key][key2][key3] = aligned_model[key][key2][key3]
            self.aligned_model = aligned_dic
            return self.aligned_model
        else:
            self.aligned_model = aligned_model
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






