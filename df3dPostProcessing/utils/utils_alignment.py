import numpy as np
import cv2 as cv
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def align_data(exp_dict):
    exp_dict = default_order_of_axis(exp_dict)
    fix = fixed_lengths_and_base_point(exp_dict)
    align = align_model(fix)
    
    return align

def fixed_lengths_and_base_point(raw_dict):
    new_dict = {}
    for segment, landmarks in raw_dict.items():
        if 'leg' in segment:
            new_dict[segment] = {}
            for landmark, pos in landmarks.items():
                new_dict[segment][landmark]={}
                dist = []
                pos_t = pos.transpose()
                #pos_t[1] = -pos_t[1]
                #corr_pos = pos_t.transpose()
                corr_pos = pos
                if 'Coxa' in landmark:
                    mean_x = np.mean(pos_t[0])
                    mean_y = np.mean(pos_t[1])
                    mean_z = np.mean(pos_t[2])
                    new_dict[segment][landmark]['fixed_pos']=[mean_x,mean_y,mean_z]
                    for i, point in enumerate(corr_pos):
                        #key = [name for name in landmarks.keys() if 'Femur' in name]
                        a = point
                        b = raw_dict[segment]['Femur'][i] 
                        dist.append(np.linalg.norm(a-b))
                    
                if 'Femur' in landmark:
                    for i, point in enumerate(corr_pos):
                        a = point
                        b = raw_dict[segment]['Tibia'][i] 
                        dist.append(np.linalg.norm(a-b))

                if 'Tibia' in landmark:
                    for i, point in enumerate(corr_pos):
                        a = point
                        b = raw_dict[segment]['Tarsus'][i] 
                        dist.append(np.linalg.norm(a-b))

                if 'Tarsus' in landmark:
                    for i, point in enumerate(corr_pos):
                        a = point
                        b = raw_dict[segment]['Claw'][i] 
                        dist.append(np.linalg.norm(a-b))

                if 'Claw' in landmark:
                    new_dict[segment][landmark]['raw_pos']=pos
                    break
                    
                new_dict[segment][landmark]['raw_pos']=corr_pos
                
                new_dict[segment][landmark]['mean_length']=np.mean(dist)    
    return new_dict

def align_model(fixed_dict):
    front_coxae = []
    middle_coxae = []
    hind_coxae = []
    coxae={}
    for leg, joints in fixed_dict.items():
        for joint, data in joints.items():
            if 'F_leg' in leg and 'Coxa' in joint:
                front_coxae.append(data['fixed_pos'])
            if 'M_leg' in leg and 'Coxa' in joint:
                middle_coxae.append(data['fixed_pos'])
            if 'H_leg' in leg and 'Coxa' in joint:
                hind_coxae.append(data['fixed_pos'])

    coxae['F_'] = np.array(front_coxae)
    coxae['M_'] = np.array(middle_coxae)
    coxae['H_'] = np.array(hind_coxae)

    alignment = {}
    for pos, coords in coxae.items():
        alignment[pos] = {}
        y_angle = np.arctan2(coords[1,2]-coords[0,2],coords[1,0]-coords[0,0]) * 180 / np.pi + 90
        x_angle = (np.arctan2(coords[1,2]-coords[0,2],coords[1,1]-coords[0,1])*180/np.pi+90)*np.cos(np.deg2rad(y_angle))+90
        middle_point= [(point[0]+point[1])/2 for point in coords.transpose()]
        alignment[pos]['y_angle']= y_angle
        alignment[pos]['x_angle']= x_angle
        alignment[pos]['middle_point']= middle_point
        #print(pos, y_angle,x_angle)
        
    aligned_dict = {}
    for leg, joints in fixed_dict.items():
        aligned_dict[leg]={}
        for joint, data in joints.items():
            aligned_dict[leg][joint]={}
            for metric, coords in data.items():
                theta_y = [angle['y_angle'] for pos, angle in alignment.items() if pos in leg][0]
                theta_x = [angle['x_angle'] for pos, angle in alignment.items() if pos in leg][0]
                mid_point = [point['middle_point'] for pos, point in alignment.items() if pos in leg]
                r = R.from_euler('yx', [theta_y,-theta_x], degrees=True)
                offset_y_new = r.apply(mid_point)[0][1]
                if '_pos' in metric:
                    key = metric + '_aligned'
                    aligned_dict[leg][joint][key] = [r.apply(coords)]
                    for points in aligned_dict[leg][joint][key]:
                        if 'fixed' in metric:                            
                            points[1] = offset_y_new-points[1]
                        else:
                            for pt in points:
                                pt[1] = offset_y_new-pt[1]
                    aligned_dict[leg][joint][key] = aligned_dict[leg][joint][key][0]
                else:
                    aligned_dict[leg][joint][metric] = coords


    #plot_3d_and_2d(aligned_dict, metric = 'raw_pos_aligned')
    #plot_fixed_coxa(aligned_dict)            

    return aligned_dict


def rescale_using_2d_data(data_3d,data_2d,cams_info,exp_dir,pixelSize=[5.86e-3,5.86e-3]):
    """
    Rescale 3d data using 2d data
    """
    right_view = {}
    left_view = {}
    #front_view = {}
    for key, info in cams_info.items():
        r = R.from_dcm(info['R']) 
        th = r.as_euler('zyx', degrees=True)[1]
        if 90-th<15:
            right_view['R_points2d'] = data_2d[key-1]
            right_view['cam_id'] = key-1 
        elif 90-th>165:
            left_view['L_points2d'] = data_2d[key-1]
            left_view['cam_id'] = key-1
        #elif abs(th)+1 < 10:
        #    front_view['F_points2d'] = data_2d[key-1]
        #    front_view['cam_id'] = key-1

    #draw_legs_from_2d(right_view, exp_dir,saveimgs=True)
    #draw_legs_on_img(right_view, exp_dir)   

    for name, leg in data_3d.items():  
        for k, joints in leg.items():
            if k == 'Femur':
                prev = 'Coxa'
            if k == 'Tibia':
                prev = 'Femur'                
            if k == 'Tarsus':
                prev = 'Tibia'             
            if k == 'Claw':
                prev = 'Tarsus'

            if k != 'Coxa':
                x_3d = joints['raw_pos_aligned'].transpose()[0] 
                y_3d = joints['raw_pos_aligned'].transpose()[1] 
                z_3d = joints['raw_pos_aligned'].transpose()[2]
                x_3d_amp = np.max(x_3d)-np.min(x_3d) 
                x_3d_zero = np.mean(data_3d[name][prev]['raw_pos_aligned'].transpose()[0])
                
                y_3d_amp = np.max(y_3d)-np.min(y_3d) 
                y_3d_zero = np.mean(data_3d[name][prev]['raw_pos_aligned'].transpose()[1])

                z_3d_amp = np.max(z_3d)-np.min(z_3d)
                z_3d_zero = np.mean(data_3d[name][prev]['raw_pos_aligned'].transpose()[2])

                if 'L' in name:
                    x_2d = left_view['L_points2d'][name][k].transpose()[0]
                    z_2d = left_view['L_points2d'][name][k].transpose()[1]
                    x_2d_amp = (np.max(x_2d)-np.min(x_2d)) * pixelSize[0]
                    z_2d_amp = (np.max(z_2d)-np.min(z_2d)) * pixelSize[1]
                if 'R' in name:
                    x_2d = right_view['R_points2d'][name][k].transpose()[0]
                    z_2d = right_view['R_points2d'][name][k].transpose()[1]
                    x_2d_amp = (np.max(x_2d)-np.min(x_2d)) * pixelSize[0]
                    z_2d_amp = (np.max(z_2d)-np.min(z_2d)) * pixelSize[1]

                #print(name, k, x_2d_amp / x_3d_amp, np.mean([x_2d_amp/x_3d_amp,z_2d_amp/z_3d_amp]), z_2d_amp / z_3d_amp)
                x_3d_scaled = []
                y_3d_scaled = []
                z_3d_scaled = []
                x_factor = x_2d_amp / x_3d_amp
                y_factor = np.mean([x_2d_amp/x_3d_amp,z_2d_amp/z_3d_amp])
                z_factor = z_2d_amp / z_3d_amp
                for i in range(len(x_3d)):
                    x_3d_scaled.append(x_3d_zero + (x_3d[i] - x_3d_zero) * x_factor)
                    y_3d_scaled.append(y_3d_zero + (y_3d[i] - y_3d_zero) * y_factor)
                    z_3d_scaled.append(z_3d_zero + (z_3d[i] - z_3d_zero) * z_factor)
                
                scaled_data = np.array([x_3d_scaled,y_3d_scaled,z_3d_scaled]).transpose()

                joints['raw_pos_aligned'] = scaled_data
    
    data_3d = recalculate_lengths(data_3d)
    
    return data_3d

def recalculate_lengths(data):
    for name, leg in data.items():
        for segment, body_part in leg.items():
                dist = []
                if 'Coxa' in segment:
                    metric = 'raw_pos_aligned'
                    next_segment = 'Femur'
                    
                if 'Femur' in segment:
                    metric = 'raw_pos_aligned'
                    next_segment = 'Tibia'
                    
                if 'Tibia' in segment:
                    metric = 'raw_pos_aligned'
                    next_segment = 'Tarsus'
                    
                if 'Tarsus' in segment:
                    metric = 'raw_pos_aligned'
                    next_segment = 'Claw'
                    
                for i, point in enumerate(body_part[metric]):
                    a = point
                    b = data[name][next_segment]['raw_pos_aligned'][i] 
                    dist.append(np.linalg.norm(a-b))
                        
                body_part['mean_length']=np.mean(dist)
    return data

def draw_legs_from_2d(cam_view,exp_dir,begin=0,end=0,saveimgs = False):
    key = [k for k in cam_view.keys() if 'points2d' in k][0]
    data = cam_view[key]
    cam_id = cam_view['cam_id']
    
    colors = {'LF':(255,0,0),'LM':(0,255,0),'LH':(0,0,255),'RF':(153,76,0),'RM':(255,128,0),'RH':(255,178,102)}
    
    if end == 0:
        end = len(data['LF_leg']['Coxa'])
    
    for frame in range(begin,end):
        df3d_dir = exp_dir.find('df3d')
        folder = exp_dir[:df3d_dir]
        img_name = folder + 'camera_' + str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
        img = cv.imread(img_name)
        for leg, body_parts in data.items():
            if leg[0] == 'L' or leg[0]=='R':
                color = colors[leg[:2]]
                for segment, points in body_parts.items():
                    if segment != 'Coxa':
                        if 'Femur' in segment:
                            start_point = data[leg]['Coxa'][frame]
                            end_point = points[frame]
                        if 'Tibia' in segment:
                            start_point = data[leg]['Femur'][frame]
                            end_point = points[frame]
                        if 'Tarsus' in segment:
                            start_point = data[leg]['Tibia'][frame]
                            end_point = points[frame]
                        if 'Claw' in segment:
                            start_point = data[leg]['Tarsus'][frame]
                            end_point = points[frame]
                        img = draw_lines(img,start_point,end_point,color=color)
        if saveimgs:
            file_name = exp_dir.split('/')[-1]
            new_folder = 'results/tracking_2d/'+file_name.replace('.pkl','/')
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            name = new_folder + 'camera_' + str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
            cv.imwrite(name,img)
        cv.imshow('img',img)
        cv.waitKey(1)
    cv.destroyAllWindows()


def draw_lines(img, start, end, color = (255, 0, 0)):
    coords_prev = np.array(start).astype(int)
    coords_next = np.array(end).astype(int)

    start_point = (coords_prev[0],coords_prev[1])
    end_point = (coords_next[0],coords_next[1]) 
    thickness = 5
    
    cv.line(img, start_point, end_point, color, thickness) 

    return img


def default_order_of_axis(exp_dict): 
    """
    Reorders the and mirrors the axis to retrieve the
    original orientation as give by df3d. This is necessary
    because some versions of df3d automatically mirror and swap
    axes to make visualization easier.

    Parameters
    ----------
    exp_dict : dictionary
        This dictionary holds all the raw joint positions.

    Returns
    -------
    exp_dict : dictionary
        Dictionary with reordered axes and mirrored axes.
    """
    coxa_points = []
    claw_points = []
    for leg, leg_data in exp_dict.items():
        for joint, joint_data in leg_data.items():
            if joint == "Coxa":
                coxa_points.append(joint_data)
            if joint == "Claw":
                claw_points.append(joint_data)

    coxa_points = np.array(coxa_points)
    claw_point = np.array(claw_points)

    coxa_points_for_svd = coxa_points.reshape((-1, 3))
    centroid = np.mean(coxa_points_for_svd, axis=0)
    coxa_points_for_svd = coxa_points_for_svd - centroid
    U, S, VT = np.linalg.svd(np.transpose(coxa_points_for_svd))
    second_axis = np.argmax(np.abs(U[:, 2]))

    mirror_factors = np.ones(3)
    
    # Check for flip of second axis
    claw_points = np.mean(claw_points, axis=1)
    if np.mean(np.mean(coxa_points, axis=1)[:, second_axis] > claw_points[:, second_axis]) > 0.5:
        mirror_factors[1] = -1
    
    remaining_axes = [i for i in range(3) if i != second_axis]
    
    left_side_remaining_axis_0 = np.mean(coxa_points[(0, 1, 2), :, remaining_axes[0]], axis=1)
    right_side_remaining_axis_0 = np.mean(coxa_points[(3, 4, 5), :, remaining_axes[0]], axis=1)
    left_side_remaining_axis_1 = np.mean(coxa_points[(0, 1, 2), :, remaining_axes[1]], axis=1)
    right_side_remaining_axis_1 = np.mean(coxa_points[(3, 4, 5), :, remaining_axes[1]], axis=1)

    if (
        np.all(left_side_remaining_axis_0 > right_side_remaining_axis_0) and 
        np.all(left_side_remaining_axis_1 > right_side_remaining_axis_1)
       ):
        
        if (
            np.all(np.diff(left_side_remaining_axis_0) < 0) and 
            np.all(np.diff(left_side_remaining_axis_1) > 0) and 
            np.all(np.diff(right_side_remaining_axis_0) < 0) and 
            np.all(np.diff(right_side_remaining_axis_1) > 0)
           ):
            first_axis = remaining_axes[0]
            third_axis = remaining_axes[1]
        elif (
            np.all(np.diff(left_side_remaining_axis_0) > 0) and 
            np.all(np.diff(left_side_remaining_axis_1) < 0) and 
            np.all(np.diff(right_side_remaining_axis_0) > 0) and 
            np.all(np.diff(right_side_remaining_axis_1) < 0)
           ):
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
        elif (
              np.all(np.diff(left_side_remaining_axis_0) < 0) and
              np.all(np.diff(left_side_remaining_axis_1) < 0) and 
              np.all(np.diff(right_side_remaining_axis_0) < 0) and 
              np.all(np.diff(right_side_remaining_axis_1) < 0)
             ): 
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
        else:
            raise ValueError("Can not determine correct order of axis.")
    elif (
          np.all(left_side_remaining_axis_0 < right_side_remaining_axis_0) and 
          np.all(left_side_remaining_axis_1 < right_side_remaining_axis_1)
         ):
        if (
            np.all(np.diff(left_side_remaining_axis_0) < 0) and 
            np.all(np.diff(left_side_remaining_axis_1) > 0) and 
            np.all(np.diff(right_side_remaining_axis_0) < 0) and 
            np.all(np.diff(right_side_remaining_axis_1) > 0)
           ):
            # fixed for [[ 0. -1.  0.], [ 0.  0. -1.], [-1.  0.  0.]]
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
            mirror_factors[0] = -1
            mirror_factors[2] = -1
        elif (
            np.all(np.diff(left_side_remaining_axis_0) > 0) and 
            np.all(np.diff(left_side_remaining_axis_1) < 0) and 
            np.all(np.diff(right_side_remaining_axis_0) > 0) and 
            np.all(np.diff(right_side_remaining_axis_1) < 0)
           ):
            # fixed for [[-1.  0.  0.], [ 0. -1.  0.], [ 0.  0. -1.]]
            first_axis = remaining_axes[0]
            third_axis = remaining_axes[1]
            mirror_factors[0] = -1
            mirror_factors[2] = -1
        elif (
              np.all(np.diff(left_side_remaining_axis_0) < 0) and
              np.all(np.diff(left_side_remaining_axis_1) < 0) and 
              np.all(np.diff(right_side_remaining_axis_0) < 0) and 
              np.all(np.diff(right_side_remaining_axis_1) < 0)
             ): 
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
        else:
            raise ValueError("Can not determine correct order of axis.")
    elif (
          np.all(left_side_remaining_axis_0 < right_side_remaining_axis_0) and 
          np.all(left_side_remaining_axis_1 > right_side_remaining_axis_1)
         ):
        if (
            np.all(np.diff(left_side_remaining_axis_0) < 0) and 
            np.all(np.diff(left_side_remaining_axis_1) > 0) and 
            np.all(np.diff(right_side_remaining_axis_0) < 0) and 
            np.all(np.diff(right_side_remaining_axis_1) > 0)
           ):
            first_axis = remaining_axes[0]
            third_axis = remaining_axes[1]
        elif (
            np.all(np.diff(left_side_remaining_axis_0) > 0) and 
            np.all(np.diff(left_side_remaining_axis_1) < 0) and 
            np.all(np.diff(right_side_remaining_axis_0) > 0) and 
            np.all(np.diff(right_side_remaining_axis_1) < 0)
           ):
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
        elif (
              np.all(np.diff(left_side_remaining_axis_0) < 0) and
              np.all(np.diff(left_side_remaining_axis_1) < 0) and 
              np.all(np.diff(right_side_remaining_axis_0) < 0) and 
              np.all(np.diff(right_side_remaining_axis_1) < 0)
             ): 
            # fixed for [[0. 1. 0.], [1. 0. 0.], [0. 0. 1.]]
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
            mirror_factors[2] = -1
        elif (
              np.all(np.diff(left_side_remaining_axis_0) > 0) and
              np.all(np.diff(left_side_remaining_axis_1) > 0) and 
              np.all(np.diff(right_side_remaining_axis_0) > 0) and 
              np.all(np.diff(right_side_remaining_axis_1) > 0)
             ): 
            # fixed for [[-1.  0.  0.], [ 0. -1.  0.], [ 0.  0.  1.]]
            first_axis = remaining_axes[0]
            third_axis = remaining_axes[1]
            mirror_factors[0] = -1
        else:
            raise ValueError("Can not determine correct order of axis.")
    elif (
          np.all(left_side_remaining_axis_0 > right_side_remaining_axis_0) and 
          np.all(left_side_remaining_axis_1 < right_side_remaining_axis_1)
         ):
        if (
            np.all(np.diff(left_side_remaining_axis_0) < 0) and 
            np.all(np.diff(left_side_remaining_axis_1) > 0) and 
            np.all(np.diff(right_side_remaining_axis_0) < 0) and 
            np.all(np.diff(right_side_remaining_axis_1) > 0)
           ):
            first_axis = remaining_axes[0]
            third_axis = remaining_axes[1]
        elif (
            np.all(np.diff(left_side_remaining_axis_0) > 0) and 
            np.all(np.diff(left_side_remaining_axis_1) < 0) and 
            np.all(np.diff(right_side_remaining_axis_0) > 0) and 
            np.all(np.diff(right_side_remaining_axis_1) < 0)
           ):
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
        elif (
              np.all(np.diff(left_side_remaining_axis_0) < 0) and
              np.all(np.diff(left_side_remaining_axis_1) < 0) and 
              np.all(np.diff(right_side_remaining_axis_0) < 0) and 
              np.all(np.diff(right_side_remaining_axis_1) < 0)
             ): 
            # fixed for [[ 1.  0.  0.], [ 0. -1.  0.], [ 0.  0. -1.]]
            first_axis = remaining_axes[0]
            third_axis = remaining_axes[1]
            mirror_factors[2] = -1
        elif (
              np.all(np.diff(left_side_remaining_axis_0) > 0) and
              np.all(np.diff(left_side_remaining_axis_1) > 0) and 
              np.all(np.diff(right_side_remaining_axis_0) > 0) and 
              np.all(np.diff(right_side_remaining_axis_1) > 0)
             ): 
            # fixed for [[ 0. -1.  0.], [ 0.  0. -1.], [ 1.  0.  0.]]
            first_axis = remaining_axes[1]
            third_axis = remaining_axes[0]
            mirror_factors[0] = -1
        else:
            raise ValueError("Can not determine correct order of axis.")
    else:
        raise ValueError("Cannot determine correct order of axis.")
    return _swap_axes_in_dict(exp_dict, (first_axis, second_axis, third_axis), mirror_factors)


def _swap_axes_in_dict(exp_dict, axes_order, mirror_factors=[1, 1, 1]):
    for leg, leg_data in exp_dict.items():
        for joint, joint_data in leg_data.items():
            processed_data = joint_data[:, axes_order]
            processed_data = processed_data * mirror_factors
            exp_dict[leg][joint] = processed_data
    return exp_dict
