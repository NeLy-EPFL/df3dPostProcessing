""" use this file to generate template oscillation from the pkl file """
from df3dPostProcessing import df3dPostProcess
from df3dPostProcessing.utils import utils_plots

#experiment = 'pose_result__data_paper_180918_MDN_PR_Fly1_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_PR_Fly5_004_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180921_aDN_PR_Fly8_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_CsCh_Fly6_001_SG1_behData_images.pkl'
#experiment = '/home/nely/Desktop/animationSimfly/video2/180921_aDN_PR_Fly8_005_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_PR_Fly8_005_SG1_behData_images_images.pkl'
#experiment = 'data/pose_result__home_nely_Desktop_animationSimfly_video2_180919_MDN_CsCh_Fly6_001_SG1_behData_images_images.pkl'
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
