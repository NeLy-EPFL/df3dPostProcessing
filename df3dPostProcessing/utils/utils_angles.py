import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def angle_between_segments(prev_joint, joint, next_joint, rot_axis):
    v1 = prev_joint - joint
    v2 = next_joint - joint
    #print(v1)
    #print(v2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)

    det = np.linalg.det([rot_axis,v1,v2,])
    if det < 0:
        angle_corr = -angle
    else:
        angle_corr = angle

    return angle_corr

def calculate_yaw(coxa_origin, femur_pos):
    next_joint = femur_pos.copy()
    next_joint[0] = coxa_origin[0]
    z_axis = coxa_origin + [0,0,-1]
    rot_axis = [1,0,0]

    angle = angle_between_segments(z_axis, coxa_origin, next_joint, rot_axis)

    return angle

def calculate_pitch(coxa_origin, femur_pos):
    next_joint = femur_pos.copy()
    next_joint[1] = coxa_origin[1]
    z_axis = coxa_origin + [0,0,-1]
    rot_axis = [0,1,0]
    
    angle = angle_between_segments(z_axis, coxa_origin, next_joint, rot_axis)

    return angle

def calculate_roll(coxa_origin,femur_pos,tibia_pos,r,leg):
    length = np.linalg.norm(femur_pos-coxa_origin)
    
    if 'F_' in leg:
        prev_joint = np.array([1,0,-length])
        rot_axis = [0,0,1]
    elif 'LM' in leg or 'LH' in leg:
        prev_joint = np.array([0,-1,-length])
        rot_axis = [0,0,-1]
    elif 'RM' in leg or 'RH' in leg:
        prev_joint = np.array([0,1,-length])
        rot_axis = [0,0,1]
        
    curr_joint = np.array([0,0,-length])
    r_inv = r.inv()
    tibia = tibia_pos - coxa_origin
    next_joint = r_inv.apply(tibia)
    next_joint[2] = -length
    

    angle = angle_between_segments(prev_joint,curr_joint,next_joint,rot_axis)

    return angle
    
def calculate_angles(aligned_dict,begin,end):
    angles_dict = {}
    if end == 0:
        end = len(aligned_dict['RF_leg']['Coxa']['raw_pos_aligned'])
    for leg, joints in aligned_dict.items():
        angles_dict[leg]={}
        factor_zero = -np.pi
        if 'F' in leg:
            rot_axis = [0,1,0]
        elif 'LM' in leg or 'LH' in leg:
            rot_axis = [1,0,0]
        elif 'RM' in leg or 'RH' in leg:
            rot_axis = [-1,0,0]
        for joint, data in joints.items():
            angles = []
            if 'Coxa' in joint:
                angles_dict[leg]['yaw']=[]
                angles_dict[leg]['pitch']=[]
                angles_dict[leg]['roll']=[]
                coxa_origin = data['fixed_pos_aligned']
                coxa_length = data['mean_length']
                joints['Femur']['recal_pos']=[]
                for i in range(begin,end):
                    femur_pos = joints['Femur']['raw_pos_aligned'][i]
                    yaw = calculate_yaw(coxa_origin,femur_pos)
                    pitch = calculate_pitch(coxa_origin,femur_pos)
                    tibia_pos = joints['Tibia']['raw_pos_aligned'][i]
                    r = R.from_euler('zyx', [0,pitch,yaw])
                    roll = calculate_roll(coxa_origin,femur_pos,tibia_pos,r,leg)
                    #r2 = R.from_euler('zyx', [roll,pitch,yaw])
                    #curr = [0,0,-length]
                    #joints['Femur']['recal_pos'].append(r2.apply(curr)+origin)
                    angles_dict[leg]['yaw'].append(yaw)
                    angles_dict[leg]['pitch'].append(pitch)
                    angles_dict[leg]['roll'].append(roll)
                    #print(leg)
                    #print(yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi)
            if 'Femur' in joint:
                angles_dict[leg]['th_fe']=[]
                for i in range(begin,end):
                    #origin = data['recal_pos'][i]
                    coxa_pos = joints['Coxa']['fixed_pos_aligned']
                    femur_pos = data['raw_pos_aligned'][i]
                    tibia_pos = joints['Tibia']['raw_pos_aligned'][i]                        
                    th_femur = angle_between_segments(coxa_pos, femur_pos, tibia_pos,rot_axis)
                    #print(th_femur*180/np.pi)
                    th_femur = factor_zero + th_femur
                    #print(th_femur*180/np.pi)
                    angles_dict[leg]['th_fe'].append(th_femur)
            if 'Tibia' in joint:
                angles_dict[leg]['th_ti']=[]
                for i in range(begin,end):
                    #origin = data['recal_pos'][i]
                    femur_pos = joints['Femur']['raw_pos_aligned'][i]
                    tibia_pos = data['raw_pos_aligned'][i]
                    tarsus_pos = joints['Tarsus']['raw_pos_aligned'][i]
                    th_tibia = angle_between_segments(femur_pos, tibia_pos, tarsus_pos,rot_axis)
                    #print(th_tibia*180/np.pi)
                    th_tibia = th_tibia-factor_zero
                    #print(th_tibia*180/np.pi)
                    angles_dict[leg]['th_ti'].append(th_tibia)
            if 'Tarsus' in joint:
                angles_dict[leg]['th_ta']=[]
                for i in range(begin,end):
                    #origin = data['recal_pos'][i]
                    tibia_pos = joints['Tibia']['raw_pos_aligned'][i]
                    tarsus_pos = data['raw_pos_aligned'][i]
                    claw_pos = joints['Claw']['raw_pos_aligned'][i]
                    th_tarsus = angle_between_segments(tibia_pos, tarsus_pos, claw_pos,rot_axis)
                    #print(th_tarsus*180/np.pi)
                    th_tarsus = factor_zero + th_tarsus
                    if abs(th_tarsus) > 1.7:
                        th_tarsus = -2*np.pi - th_tarsus
                    #print(th_tarsus*180/np.pi)
                    angles_dict[leg]['th_ta'].append(th_tarsus)

    return angles_dict


def calculate_forward_kinematics(leg_name, frame, leg_angles, data_dict, roll_tr = 0):
    
    if 'LF' in leg_name or 'RF' in leg_name:
        roll = leg_angles['roll'][frame]
    elif 'LM' in leg_name or 'LH' in leg_name:
        roll = - (np.pi/2 + leg_angles['roll'][frame])
    elif 'RM' in leg_name or 'RH' in leg_name:
        roll = np.pi/2 + leg_angles['roll'][frame]
    
    r1 = R.from_euler('zyx',[roll,leg_angles['pitch'][frame],leg_angles['yaw'][frame]])
    r2 = R.from_euler('zyx',[roll_tr,leg_angles['th_fe'][frame],0])
    r3 = R.from_euler('y',leg_angles['th_ti'][frame])
    r4 = R.from_euler('y',leg_angles['th_ta'][frame])

    coxa_pos = data_dict[leg_name]['Coxa']['fixed_pos_aligned']    
    l_coxa = data_dict[leg_name]['Coxa']['mean_length']#np.linalg.norm(coxa_pos-real_pos_femur)#
    l_femur = data_dict[leg_name]['Femur']['mean_length']#np.linalg.norm(real_pos_femur-real_pos_tibia)#
    l_tibia = data_dict[leg_name]['Tibia']['mean_length']#np.linalg.norm(real_pos_tibia-real_pos_tarsus)#
    l_tarsus = data_dict[leg_name]['Tarsus']['mean_length']#np.linalg.norm(real_pos_tarsus-real_pos_claw)#

    fe_init_pos = np.array([0,0,-l_coxa])
    ti_init_pos = np.array([0,0,-l_femur])
    ta_init_pos = np.array([0,0,-l_tibia])
    claw_init_pos = np.array([0,0,-l_tarsus])

    femur_pos = r1.apply(fe_init_pos) + coxa_pos
    tibia_pos = r1.apply(r2.apply(ti_init_pos)) + femur_pos            
    tarsus_pos = r1.apply(r2.apply(r3.apply(ta_init_pos))) + tibia_pos
    claw_pos = r1.apply(r2.apply(r3.apply(r4.apply(claw_init_pos)))) + tarsus_pos

    fk_pos = np.array([coxa_pos,femur_pos,tibia_pos,tarsus_pos,claw_pos])

    return fk_pos

def calculate_best_roll_tr(angles,data_dict,begin=0,end=0):

    diff_dict = {}

    if end == 0:
        end = len(angles['LF_leg']['yaw'])    

    for frame in range(begin, end):
        print('\rFrame: '+str(frame),end='')

        for name, leg in angles.items():

            if not name in diff_dict.keys():
                diff_dict[name]={'min_dist':[],'best_roll':[]}

            coxa_pos = data_dict[name]['Coxa']['fixed_pos_aligned']
            real_pos_femur = data_dict[name]['Femur']['raw_pos_aligned'][frame]
            real_pos_tibia = data_dict[name]['Tibia']['raw_pos_aligned'][frame]
            real_pos_tarsus = data_dict[name]['Tarsus']['raw_pos_aligned'][frame]
            real_pos_claw = data_dict[name]['Claw']['raw_pos_aligned'][frame]

            min_dist = 100000000
            best_roll = 0
            for i in range(0, 500):
                if 'L' in name:
                    roll_tr = np.deg2rad(-i/10)
                elif 'R' in name:
                    roll_tr = np.deg2rad(i/10)
                    
                pos_3d = calculate_forward_kinematics(name, frame, leg, data_dict,roll_tr=roll_tr)

                dist = np.linalg.norm(pos_3d[4]-real_pos_tarsus)

                if dist<min_dist:
                    min_dist = dist
                    best_roll = roll_tr

            diff_dict[name]['min_dist'].append(min_dist)
            diff_dict[name]['best_roll'].append(best_roll)

    return diff_dict
