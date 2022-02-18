""" use this file to generate template oscillation from the pkl file """
import os
import matplotlib.pyplot as plt
from df3dPostProcessing import df3dPostProcess

directory = '/mnt/nas/GO/7cam/210805_aJO-GAL4xUAS-CsChr/Fly001/001_High/behData/images/df3d'
file_name = 'pose_result__mnt_nas_GO_7cam_210805_aJO-GAL4xUAS-CsChr_Fly001_001_High_behData_images.pkl'

experiment = os.path.join(directory, file_name)

df3d = df3dPostProcess(
    exp_dir=experiment,
    skeleton='df3d',
    calculate_3d=True,
    outlier_correction=True
    )

aligned_data_corrected = df3d.align_to_template(interpolate=False, smoothing=False)


df3d_no_correction = df3dPostProcess(
    exp_dir=experiment,
    skeleton='df3d',
    calculate_3d=True,
    outlier_correction=False
    )

aligned_data_wo_correction = df3d_no_correction.align_to_template(interpolate=False, smoothing=False)


fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(5,6), sharex=True)

leg = 'LF_leg'

for joint in aligned_data_corrected[leg]:
    ax1.plot(aligned_data_corrected[leg][joint]['raw_pos_aligned'][:,0], label=f'{joint} corrected')
    ax1.plot(aligned_data_wo_correction[leg][joint]['raw_pos_aligned'][:,0], label=f'{joint} not corrected')
    ax1.set_ylabel('x axis')

    ax2.plot(aligned_data_corrected[leg][joint]['raw_pos_aligned'][:,1], label=f'{joint} corrected')
    ax2.plot(aligned_data_wo_correction[leg][joint]['raw_pos_aligned'][:,1], label=f'{joint} not corrected')
    ax2.set_ylabel('y axis')

    ax3.plot(aligned_data_corrected[leg][joint]['raw_pos_aligned'][:,2], label=f'{joint} corrected')
    ax3.plot(aligned_data_wo_correction[leg][joint]['raw_pos_aligned'][:,2], label=f'{joint} not corrected')
    ax3.set_ylabel('z axis')
    ax3.set_xlabel('Frames')

plt.xlim(1600,1750)
plt.suptitle('Left Front Leg')
plt.legend()
plt.tight_layout()
plt.show()
