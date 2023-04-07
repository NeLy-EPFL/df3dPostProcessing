""" use this file to generate template oscillation from the pkl file """
from df3dPostProcessing import df3dPostProcess
from df3dPostProcessing.utils import utils_plots
# Added by Ari:
import pose3d_flytracker_utils as pfu
import pandas as pd
from pathlib import Path



# code bit - remove once you've sorted out locations

FT_DATA = Path(
    "/home/aalonso/Documents/Term2/try_git/df3dPostProcessing/examples/ft_data/" # Go into sibling directory data and into folder 'nmf_ ...'
)

xyz_file_path = FT_DATA /"input_data"/ "joints3d_xyz.csv"
sorted_df = pd.read_csv(xyz_file_path).sort_values(by=['Frame_idx'], ascending=True).reset_index(drop=True) # df containing raw, cartesian ijk points
filled_df = pfu.fill_missing_joints(sorted_df).dropna().reset_index(drop = True)

ft_3d_pose_file = str(FT_DATA/"ft_3d_array.pkl") # as the PostProcessing func deals with strings

pose_array, frames = pfu.df_to_array(sorted_df, frames = True, save_array=True, save_as=ft_3d_pose_file)


df3d = df3dPostProcess(ft_3d_pose_file, skeleton = 'flytracker')

align = df3d.align_to_template()

angles = df3d.calculate_leg_angles(save_angles=True)

dense_angles = pfu.sparse_to_dense_angles(angles, frames, save_dense_angles = True, save_as=str(FT_DATA/"dense_angles.pkl"))

velocities = df3d.calculate_velocity(angles,
                                     window=11,
                                     order=3, 
                                     time_step=0.001, 
                                     save_velocities=True)

utils_plots.plot_legs_from_angles(
        angles = angles,
        data_dict=align,
        exp_dir = 'flytracker_data',
        begin=0,
        end=0,
        plane='xz',
        saveImgs=False,
        dir_name='km',
        extraDOF={},
        ik_angles=False,
        pause=False,
        lim_axes=True)

print('hi')


# THEN put through Victor's thing to get a visualisation
# THEN put through victor's thing to get velocities

#experiment = 'pose_result__data_paper_180918_MDN_PR_Fly1_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_PR_Fly5_004_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180921_aDN_PR_Fly8_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_CsCh_Fly6_001_SG1_behData_images.pkl'
#experiment = '/home/nely/Desktop/animationSimfly/video2/180921_aDN_PR_Fly8_005_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_PR_Fly8_005_SG1_behData_images_images.pkl'
#experiment = 'data/pose_result__home_nely_Desktop_animationSimfly_video2_180919_MDN_CsCh_Fly6_001_SG1_behData_images_images.pkl'

"""
ft_experiment = '/home/aalonso/Documents/Term2/try_git/df3dPostProcessing/examples/ft_data/flytracker_raw3d_array.pkl'

df3d = df3dPostProcess(ft_experiment, skeleton = 'flytracker')

align = df3d.align_to_template()

angles = df3d.calculate_leg_angles(save_angles=True)

velocities = df3d.calculate_velocity(angles,
                                     window=11,
                                     order=3, 
                                     time_step=0.001, 
                                     save_velocities=True)


utils_plots.plot_legs_from_angles(
        angles = angles,
        data_dict=align,
        exp_dir = 'flytracker_data',
        begin=0,
        end=0,
        plane='xz',
        saveImgs=False,
        dir_name='km',
        extraDOF={},
        ik_angles=False,
        pause=False,
        lim_axes=True)

print('hi')
"""
