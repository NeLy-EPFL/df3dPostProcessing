""" use this file to generate template oscillation from the pkl file """
import numpy as np
import os
import math
import matplotlib.pyplot as plt  
from utils.angle_util import *
from utils.utils import *

#experiment = 'pose_result__data_paper_180918_MDN_PR_Fly1_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_PR_Fly5_004_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180921_aDN_PR_Fly8_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_CsCh_Fly6_001_SG1_behData_images.pkl'
experiment = '../deepfly3d_utils/data/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_PR_Fly8_005_SG1_behData_images_images.pkl'
#experiment = '../deepfly3d_utils/data/pose_result__home_nely_Desktop_animationSimfly_video2_180919_MDN_CsCh_Fly6_001_SG1_behData_images_images.pkl'



#exp_data = load_data(experiment, multiple = True, file_name = '../deepfly3d_utils/data/pose_result_smooth.pkl')
#exp_data = load_data(experiment, multiple = True, file_name = 'data/pose_result_smooth.pkl')
exp_data = load_data(experiment)

exp_dict = load_data_to_dict(exp_data)

fix = fixed_lengths_and_base_point(exp_dict)

align = align_model(fix)

angles = calculate_angles(align)
'''
times = np.arange(0,fly_angles.shape[0],1)
for jointNum in range(30,33):
    y = fly_angles[:,jointNum]
    plt.plot(times,y)

plt.show()
'''
