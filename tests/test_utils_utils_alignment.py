import numpy as np

import df3dPostProcessing.df3dPostProcessing
import df3dPostProcessing.utils.utils_alignment

def test_default_order_of_axis():
    all_possible_transforms = []
    for first_axis in range(3):
        for first_sign in [-1, 1]:
            for second_axis in range(3):
                if second_axis == first_axis:
                    continue
                for second_sign in [-1, 1]:
                    for third_axis in range(3):
                        for third_sign in [-1, 1]:
                            if third_axis == first_axis or third_axis == second_axis:
                                continue
                            transform = np.zeros((3, 3))
                            transform[0, first_axis] = first_sign
                            transform[1, second_axis] = second_sign
                            transform[2, third_axis] = third_sign
                            all_possible_transforms.append(transform)
    
    raw_data = np.load("data/pose_result__mnt_NAS_CLC_181125_R85A11-tdTomGC6fopt_Fly1_CO2xzGG_behData_001_images.pkl", allow_pickle=True)["points3d"]
    exp_dict = df3dPostProcessing.df3dPostProcessing.load_data_to_dict(raw_data)
    
    for transform in all_possible_transforms:
        transformed_raw_data = np.tensordot(raw_data, transform, axes=([2],[0]))
        transformed_exp_dict = df3dPostProcessing.df3dPostProcessing.load_data_to_dict(transformed_raw_data)
        result_exp_dict = df3dPostProcessing.utils.utils_alignment.default_order_of_axis(transformed_exp_dict)
        for leg, leg_data in exp_dict.items():
            for joint, joint_data in leg_data.items():
                assert np.allclose(joint_data, result_exp_dict[leg][joint])
