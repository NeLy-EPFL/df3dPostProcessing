import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def angle_between_segments(prev_joint, joint, next_joint, rot_axis):
    v1 = prev_joint - joint
    v2 = next_joint - joint
    # print(v1)
    # print(v2)
    try:
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    except BaseException:
        cos_angle = 0
    angle = np.arccos(cos_angle)

    det = np.linalg.det([rot_axis, v1, v2, ])
    if det < 0:
        angle_corr = -angle
    else:
        angle_corr = angle

    return angle_corr


def calculate_yaw(coxa_origin, femur_pos):
    next_joint = femur_pos.copy()
    next_joint[0] = coxa_origin[0]
    z_axis = coxa_origin + [0, 0, -1]
    rot_axis = [1, 0, 0]

    angle = angle_between_segments(z_axis, coxa_origin, next_joint, rot_axis)

    return angle


def calculate_pitch(coxa_origin, femur_pos):
    next_joint = femur_pos.copy()
    next_joint[1] = coxa_origin[1]
    z_axis = coxa_origin + [0, 0, -1]
    rot_axis = [0, 1, 0]

    angle = angle_between_segments(z_axis, coxa_origin, next_joint, rot_axis)

    return angle


def calculate_roll(coxa_origin, femur_pos, tibia_pos, r, leg):
    #length = np.linalg.norm(femur_pos-coxa_origin)

    if 'RF' in leg:
        prev_joint = np.array([1, 0, 0])
        #rot_axis = [0,0,1]
    elif 'RM' in leg:
        prev_joint = np.array([0, -1, 0])
        #rot_axis = [0,0,1]
    elif 'RH' in leg:
        prev_joint = np.array([-1, 0, 0])
        #rot_axis = [0,0,1]
    elif 'LF' in leg:
        prev_joint = np.array([1, 0, 0])
        #rot_axis = [0,0,1]
    elif 'LM' in leg:
        prev_joint = np.array([0, 1, 0])
        #rot_axis = [0,0,1]
    elif 'LH' in leg:
        prev_joint = np.array([-1, 0, 0])

    rot_axis = [0, 0, 1]

    curr_joint = np.array([0, 0, 0])
    r_inv = r.inv()
    tibia = tibia_pos - coxa_origin
    next_joint = r_inv.apply(tibia)
    next_joint[2] = 0

    angle = angle_between_segments(
        prev_joint, curr_joint, next_joint, rot_axis)

    # if 'LF' in leg and angle<-np.pi/2:
    #    angle = np.deg2rad(-110)
    # elif 'RF' in leg and angle>np.pi/2:
    #    angle -= np.pi/2

    return angle


def calculate_roll_trochanter(leg_name, angles, data_dict, frame, zero_pose):
    leg_angles = angles[leg_name]

    if 'RF' in leg_name:
        rot_axis = [-1, 0, 0]
    elif 'RM' in leg_name:
        rot_axis = [0, 1, 0]
    elif 'RH' in leg_name:
        rot_axis = [1, 0, 0]
    elif 'LF' in leg_name:
        rot_axis = [-1, 0, 0]
    elif 'LM' in leg_name:
        rot_axis = [0, -1, 0]
    elif 'LH' in leg_name:
        rot_axis = [1, 0, 0]

    yaw = leg_angles['ThC_yaw'][frame] + zero_pose[leg_name]['ThC_yaw']
    pitch = leg_angles['ThC_pitch'][frame] + zero_pose[leg_name]['ThC_pitch']
    roll = leg_angles['ThC_roll'][frame] + zero_pose[leg_name]['ThC_roll']
    th_fe = leg_angles['CTr_pitch'][frame] + zero_pose[leg_name]['CTr_pitch']
    th_ti = leg_angles['FTi_pitch'][frame] + zero_pose[leg_name]['FTi_pitch']

    r1 = R.from_euler('zyx', [roll, pitch, yaw])
    r2 = R.from_euler('zyx', [0, th_fe, 0])
    r3 = R.from_euler('y', th_ti)

    coxa_pos = data_dict['Coxa']['fixed_pos_aligned']
    l_coxa = data_dict['Coxa']['mean_length']
    l_femur = data_dict['Femur']['mean_length']
    l_tibia = data_dict['Tibia']['mean_length']

    real_pos_femur = data_dict['Femur']['raw_pos_aligned'][frame]
    real_pos_tibia = data_dict['Tibia']['raw_pos_aligned'][frame]
    real_pos_tarsus = data_dict['Tarsus']['raw_pos_aligned'][frame]

    fe_init_pos = np.array([0, 0, -l_coxa])
    ti_init_pos = np.array([0, 0, -l_femur])
    ta_init_pos = np.array([0, 0, -l_tibia])

    femur_pos = r1.apply(fe_init_pos) + coxa_pos
    tibia_pos = r1.apply(r2.apply(ti_init_pos)) + real_pos_femur
    tarsus_pos = r1.apply(r2.apply(r3.apply(ta_init_pos))) + real_pos_tibia

    angle = angle_between_segments(
        tarsus_pos,
        real_pos_tibia,
        real_pos_tarsus,
        rot_axis)

    return angle


def calculate_angles(aligned_dict, begin, end, get_CTr_roll, zero_pose):
    angles_dict = {}
    if end == 0:
        end = len(aligned_dict['RF_leg']['Coxa']['raw_pos_aligned'])
    for leg, joints in aligned_dict.items():
        angles_dict[leg] = {}
        if 'F' in leg:
            flex_axis = [0, 1, 0]
        elif 'RM' in leg:
            flex_axis = [1, 0, 0]
        elif 'RH' in leg:
            flex_axis = [0, -1, 0]
        elif 'LM' in leg:
            flex_axis = [-1, 0, 0]
        elif 'LH' in leg:
            flex_axis = [0, -1, 0]
        for joint, data in joints.items():
            angles = []
            if 'Coxa' in joint:
                angles_dict[leg]['ThC_yaw'] = []
                angles_dict[leg]['ThC_pitch'] = []
                angles_dict[leg]['ThC_roll'] = []
                coxa_origin = data['fixed_pos_aligned']
                coxa_length = data['mean_length']
                joints['Femur']['recal_pos'] = []
                for i in range(begin, end):
                    femur_pos = joints['Femur']['raw_pos_aligned'][i]
                    yaw = calculate_yaw(coxa_origin, femur_pos)
                    pitch = calculate_pitch(coxa_origin, femur_pos)
                    tibia_pos = joints['Tibia']['raw_pos_aligned'][i]
                    r = R.from_euler('zyx', [0, pitch, yaw])
                    roll = calculate_roll(
                        coxa_origin, femur_pos, tibia_pos, r, leg)

                    angles_dict[leg]['ThC_yaw'].append(yaw)
                    angles_dict[leg]['ThC_pitch'].append(pitch)
                    angles_dict[leg]['ThC_roll'].append(roll)

            if 'Femur' in joint:
                angles_dict[leg]['CTr_pitch'] = []
                for i in range(begin, end):
                    #origin = data['recal_pos'][i]
                    coxa_pos = joints['Coxa']['fixed_pos_aligned']
                    femur_pos = data['raw_pos_aligned'][i]
                    tibia_pos = joints['Tibia']['raw_pos_aligned'][i]
                    th_femur = angle_between_segments(
                        coxa_pos, femur_pos, tibia_pos, flex_axis)
                    if th_femur < 0:
                        th_femur = -th_femur
                    angles_dict[leg]['CTr_pitch'].append(th_femur)

            if 'Tibia' in joint:
                angles_dict[leg]['FTi_pitch'] = []
                if get_CTr_roll:
                    angles_dict[leg]['CTr_roll'] = []
                for i in range(begin, end):
                    #origin = data['recal_pos'][i]
                    femur_pos = joints['Femur']['raw_pos_aligned'][i]
                    tibia_pos = data['raw_pos_aligned'][i]
                    tarsus_pos = joints['Tarsus']['raw_pos_aligned'][i]
                    th_tibia = angle_between_segments(
                        femur_pos, tibia_pos, tarsus_pos, flex_axis)
                    if th_tibia > 0:
                        th_tibia = -th_tibia
                    angles_dict[leg]['FTi_pitch'].append(th_tibia)

                    if get_CTr_roll:
                        roll_tr = calculate_roll_trochanter(
                            leg, angles_dict, joints, i - begin, zero_pose)
                        if ('RF' in leg and roll_tr > 0) or (
                                'LF' in leg and roll_tr < 0):
                            roll_tr = -roll_tr
                        elif ('RM' in leg and roll_tr > 0) or ('LM' in leg and roll_tr < 0):
                            roll_tr = -roll_tr
                        # or ('LH' in leg and roll_tr>0):
                        elif ('RH' in leg and roll_tr < 0):
                            roll_tr = -roll_tr
                        angles_dict[leg]['CTr_roll'].append(roll_tr)

            if 'Tarsus' in joint:
                angles_dict[leg]['TiTa_pitch'] = []
                for i in range(begin, end):
                    tibia_pos = joints['Tibia']['raw_pos_aligned'][i]
                    tarsus_pos = data['raw_pos_aligned'][i]
                    claw_pos = joints['Claw']['raw_pos_aligned'][i]
                    th_tarsus = angle_between_segments(
                        tibia_pos, tarsus_pos, claw_pos, flex_axis)
                    if th_tarsus < 0:
                        th_tarsus = -th_tarsus
                    angles_dict[leg]['TiTa_pitch'].append(th_tarsus)

    return angles_dict


def calculate_forward_kinematics(
        leg_name,
        frame,
        leg_angles,
        data_dict,
        extraDOF={}):

    yaw = leg_angles['ThC_yaw'][frame]
    pitch = leg_angles['ThC_pitch'][frame]
    roll = leg_angles['ThC_roll'][frame]
    th_fe = leg_angles['CTr_pitch'][frame]
    th_ti = leg_angles['FTi_pitch'][frame]
    th_ta = leg_angles['TiTa_pitch'][frame]

    roll_tr = 0
    yaw_tr = 0
    roll_ti = 0
    yaw_ti = 0
    roll_ta = 0
    yaw_ta = 0

    if extraDOF:
        for key, val in extraDOF.items():
            if key == 'CTr_roll':
                roll_tr = val
            if key == 'CTr_yaw':
                yaw_tr = val
            if key == 'FTi_roll':
                roll_ti = val
            if key == 'FTi_yaw':
                yaw_ti = val
            if key == 'TiTa_roll':
                roll_ta = val
            if key == 'TiTa_yaw':
                yaw_ta = val

    r1 = R.from_euler('zyx', [roll, pitch, yaw])
    r2 = R.from_euler('zyx', [roll_tr, th_fe, yaw_tr])
    r3 = R.from_euler('zyx', [roll_ti, th_ti, yaw_ti])
    r4 = R.from_euler('zyx', [roll_ta, th_ta, yaw_ta])

    coxa_pos = data_dict[leg_name]['Coxa']['fixed_pos_aligned']
    # np.linalg.norm(coxa_pos-real_pos_femur)#
    l_coxa = data_dict[leg_name]['Coxa']['mean_length']
    # np.linalg.norm(real_pos_femur-real_pos_tibia)#
    l_femur = data_dict[leg_name]['Femur']['mean_length']
    # np.linalg.norm(real_pos_tibia-real_pos_tarsus)#
    l_tibia = data_dict[leg_name]['Tibia']['mean_length']
    # np.linalg.norm(real_pos_tarsus-real_pos_claw)#
    l_tarsus = data_dict[leg_name]['Tarsus']['mean_length']

    fe_init_pos = np.array([0, 0, -l_coxa])
    ti_init_pos = np.array([0, 0, -l_femur])
    ta_init_pos = np.array([0, 0, -l_tibia])
    claw_init_pos = np.array([0, 0, -l_tarsus])

    femur_pos = r1.apply(fe_init_pos) + coxa_pos
    tibia_pos = r1.apply(r2.apply(ti_init_pos)) + femur_pos
    tarsus_pos = r1.apply(r2.apply(r3.apply(ta_init_pos))) + tibia_pos
    claw_pos = r1.apply(
        r2.apply(
            r3.apply(
                r4.apply(claw_init_pos)))) + tarsus_pos

    fk_pos = np.array([coxa_pos, femur_pos, tibia_pos, tarsus_pos, claw_pos])

    return fk_pos


'''
def calculate_best_roll_tr(angles,data_dict,begin=0,end=0):

    diff_dict = {}

    if end == 0:
        end = len(angles['LF_leg']['yawn'])

    for frame in range(begin, end):
        print('\rFrame: '+str(frame),end='')

        for name, leg in angles.items():

            if not name in diff_dict.keys():
                diff_dict[name]={'min_dist':[],'best_roll':[]}

            #coxa_pos = data_dict[name]['Coxa']['fixed_pos_aligned']
            #real_pos_femur = data_dict[name]['Femur']['raw_pos_aligned'][frame]
            #real_pos_tibia = data_dict[name]['Tibia']['raw_pos_aligned'][frame]
            real_pos_tarsus = data_dict[name]['Tarsus']['raw_pos_aligned'][frame]
            #real_pos_claw = data_dict[name]['Claw']['raw_pos_aligned'][frame]


            min_dist = 100000000
            best_roll = 0
            for i in range(-900, 900):
                roll_tr = np.deg2rad(i/10)

                pos_3d = calculate_forward_kinematics(name, frame, leg, data_dict,roll_tr=roll_tr)

                dist = np.linalg.norm(pos_3d[4]-real_pos_tarsus)

                if dist<min_dist:
                    min_dist = dist
                    best_roll = roll_tr

            diff_dict[name]['min_dist'].append(min_dist)
            diff_dict[name]['best_roll'].append(best_roll)


    return diff_dict
'''
