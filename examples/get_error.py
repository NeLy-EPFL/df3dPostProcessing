from df3dPostProcessing import df3dPostProcess
from df3dPostProcessing.utils import utils_plots
import numpy as np
import pickle

leg_name = ['walking','grooming']

experiments = ['/home/lobato/NeuroMechFly/data/joint_tracking/walking/df3d/pose_result__210902_PR_Fly4_004_beh_behData_images.pkl',
               '/home/lobato/NeuroMechFly/data/joint_tracking/grooming/df3d/pose_result__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl']

#leg_name = ['grooming']

#experiments = ['/home/lobato/Desktop/DF3D_data/180921_aDN_CsCh_Fly6_003_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_DF3D_data_180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl']

for i, experiment in enumerate(experiments):

    df3d = df3dPostProcess(experiment)
    align = df3d.align_to_template(scale='local')
    angles = df3d.calculate_leg_angles()

    angles_ik = utils_plots.calculate_inverse_kinematics(align, init_angles=angles, roll_tr=False)

    errors = utils_plots.calculate_min_error(angles,align,extraDOF=['base','CTr_roll','CTr_yaw','FTi_roll','FTi_yaw','TiTa_roll','TiTa_yaw'])

    filename= 'errors_allExtraDOF_' + leg_name[i] + '.pkl'
    with open(filename, 'wb') as handle: 
        pickle.dump(errors, handle)
        #errors = pickle.load(handle)
        
    errors_ik = utils_plots.calculate_min_error(angles_ik,align,extraDOF=['IK'])

    filename= 'errors_IK_' + leg_name[i] + '.pkl'
    with open(filename, 'wb') as handle: 
        pickle.dump(errors_ik, handle)

    for name, leg in errors_ik.items():
        errors[name]['IK'] = errors_ik[name]['IK']

    order = ['base','IK','CTr_roll','CTr_yaw','FTi_roll','FTi_yaw','TiTa_roll','TiTa_yaw']
    errors_ordered={}
    for name,leg in errors.items():
        errors_ordered[name]={}
        for angle in order:
            errors_ordered[name][angle]=leg[angle]

    filename= 'errors_' + leg_name[i] + '_complete.pkl'
    with open(filename, 'wb') as handle: 
        pickle.dump(errors_ordered, handle)
