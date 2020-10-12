import numpy as np
import cv2 as cv
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def align_data(exp_dict,skeleton='df3d'):
    if skeleton == 'df3d':
        exp_dict = default_order_of_axis(exp_dict)
    fix = fixed_lengths_and_base_point(exp_dict)
    align = align_model(fix,skeleton)
    
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

def align_model(fixed_dict,skeleton):
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
        middle_point= [(point[0]+point[1])/2 for point in coords.transpose()]
        y_angle = np.arctan2(middle_point[2],middle_point[0])
        r_mp = R.from_euler('zyx', [0,y_angle,np.pi/2])
        new_mid_pnt = r_mp.apply(middle_point)
        #new_mid_pnt[2] *= -1
        zero_coords = coords - middle_point
        th_y = np.arctan2(zero_coords[0][0],zero_coords[0][2])
        r_roty = R.from_euler('zyx', [0,-th_y,0])
        new_coords = r_roty.apply(zero_coords)
        th_x = np.arctan2(new_coords[0][1],new_coords[0][2])        
        alignment[pos]['th_y'] = th_y
        alignment[pos]['th_x'] = th_x
        alignment[pos]['mid_pnt'] = middle_point
        alignment[pos]['offset'] = new_mid_pnt
                
    aligned_dict = {}
    for leg, joints in fixed_dict.items():
        aligned_dict[leg]={}
        theta_y = [angle['th_y'] for pos, angle in alignment.items() if pos in leg][0]
        theta_x = [angle['th_x'] for pos, angle in alignment.items() if pos in leg][0]
        mid_point = [point['mid_pnt'] for pos, point in alignment.items() if pos in leg][0]
        offset = [point['offset'] for pos, point in alignment.items() if pos in leg][0]        
        for joint, data in joints.items():
            aligned_dict[leg][joint]={}
            for metric, coords in data.items():
                if '_pos' in metric:
                    key = metric + '_aligned'
                    r = R.from_euler('zyx', [0,-theta_y,theta_x + np.pi/2])
                    zero_cent = coords - np.array(mid_point)
                    rot_coords = r.apply(zero_cent)
                    trans_coords = rot_coords + (offset - alignment['M_']['offset'])
                    align_coords = np.array([trans_coords.transpose()[0],trans_coords.transpose()[1],-trans_coords.transpose()[2]]).transpose()
                    aligned_dict[leg][joint][key] = align_coords
                    #if joint == 'Coxa':
                    #    aligned_dict[leg][joint]['offset'] = alignment['M_']['offset']*[1,1,-1]
                else:
                    aligned_dict[leg][joint][metric] = coords
       
    return aligned_dict

def rescale_using_2d_data(data_3d,data_2d,cams_info,exp_dir,pixelSize=[5.86e-3,5.86e-3],scale_procrustes = True,procrustes_factor={'LF':0.75,'LM':0.75,'LH':0.75,'RF':0.7,'RM':0.8,'RH':0.8}):
    """
    Rescale 3d data using 2d data
    """
    views = {}

    ##original: 0.8,0.25,0.4,0.2
    ##for walking: 0.75,0.1,0.15,0.0
    
    x_factor = 5.0
    y_factor = 0.35
    z_factor = -0.1

    for key, info in cams_info.items():
        r = R.from_dcm(info['R']) 
        th = r.as_euler('zyx', degrees=True)[1]
        if 90-th<15:
            views['R_points2d'] = data_2d[key-1]
            views['R_camID'] = key-1 
        elif 90-th>165:
            views['L_points2d'] = data_2d[key-1]
            views['L_camID'] = key-1

    for name, leg in data_3d.items():  
        for k, joints in leg.items():
            dist_px = views[name[:1]+'_points2d'][name][k][0]-np.mean(views[name[:1]+'_points2d'][name[:1]+'M_leg']['Coxa'],axis=0)
            #if scale_procrustes:
            #    y_dist = procrustes_factor*np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
            #else:
            #    y_dist = 0#np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
                
            dist_mm = dist_px *pixelSize
            
            #if k == 'Tarsus' and 'F' in name:
            #    y_offset = 0.1#0.5
            #elif k == 'Claw' and 'F' in name:
            #    y_offset = 0.15#0.55
            #elif k == 'Tibia' and 'F' in name:
            #    y_offset = 0.0#0.05

            if name[:1] == 'L':
                dist = np.array([dist_mm[0],0,-dist_mm[1]])
            if name[:1] == 'R':
                dist = np.array([-dist_mm[0],0,-dist_mm[1]])

            offset = joints['raw_pos_aligned'][0]-dist
            
            #x_vals = joints['raw_pos_aligned'].transpose()[0] - offset[0]
            #y_vals = joints['raw_pos_aligned'].transpose()[1] - y_offset
            #z_vals = joints['raw_pos_aligned'].transpose()[2] - offset[2]
            x_vals=[]
            y_vals=[]
            z_vals=[]
            #print(name,k,np.min(np.array(joints['raw_pos_aligned']).transpose()[1]),np.max(np.array(joints['raw_pos_aligned']).transpose()[1]))
            mean_x = np.mean(np.array(joints['raw_pos_aligned']).transpose()[0])
            diff_x = np.max(np.array(joints['raw_pos_aligned']).transpose()[0])-np.min(np.array(joints['raw_pos_aligned']).transpose()[0])
            for i, pnt in enumerate(joints['raw_pos_aligned']):
                if scale_procrustes:
                    if k == 'Coxa':
                        pnt[1] *= 0.5
                    else:
                        pnt[1] *= procrustes_factor[name[:2]]
                        
                if k!='Coxa' or k!='Femur':
                    x_new = pnt[0] - offset[0]-((pnt[0]-dist[0])/x_factor)
                else:
                    x_new = pnt[0] - offset[0]

                if abs(leg['Tarsus']['raw_pos_aligned'][i][1])<0.30 and (k=='Claw' or k=='Tarsus') and 'F' in name:
                    if k=='Claw':
                        y_factor_mod = 1.4*y_factor
                    if k=='Tarsus':
                        y_factor_mod = y_factor
                    if 'L' in name:
                        y_factor_mod *= -1
                    
                    y_new = pnt[1] - y_factor_mod
                else:
                    y_new = pnt[1]

                if k=='Claw' or k=='Tarsus':
                    z_new = pnt[2] - offset[2]*(z_factor-pnt[2]+ offset[2])/(z_factor-dist[2])
                else:
                    z_new = pnt[2] - offset[2]
                    
                x_vals.append(x_new)
                y_vals.append(y_new)
                z_vals.append(z_new)
                       
            joints['raw_pos_aligned'] = np.array([x_vals,y_vals,z_vals]).transpose()

            if k == 'Coxa':
                joints['fixed_pos_aligned'] = np.mean(joints['raw_pos_aligned'],axis=0)
    
    data_3d = recalculate_lengths(data_3d)
    
    return data_3d


'''
def rescale_using_2d_data(data_3d,data_2d,cams_info,exp_dir,pixelSize=[5.86e-3,5.86e-3],scale_procrustes = True,procrustes_factor=0.75):
    """
    Rescale 3d data using 2d data
    """
    views = {}

    ##original: 0.8,0.25,0.4,0.2
    ##for walking: 0.75,0.1,0.15,0.0
    
    for key, info in cams_info.items():
        r = R.from_dcm(info['R']) 
        th = r.as_euler('zyx', degrees=True)[1]
        if 90-th<15:
            views['R_points2d'] = data_2d[key-1]
            views['R_camID'] = key-1 
        elif 90-th>165:
            views['L_points2d'] = data_2d[key-1]
            views['L_camID'] = key-1

    for name, leg in data_3d.items():  
        for k, joints in leg.items():
            dist_px = views[name[:1]+'_points2d'][name][k][0]-views[name[:1]+'_points2d'][name[:1]+'M_leg']['Coxa'][0]
            #if scale_procrustes:
            #    y_dist = procrustes_factor*np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
            #else:
            #    y_dist = 0#np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
                
            dist_mm = dist_px *pixelSize
            y_offset = 0
            
            if k == 'Tarsus' and 'F' in name:
                y_offset = 0.1#0.5
            elif k == 'Claw' and 'F' in name:
                y_offset = 0.15#0.55
            elif k == 'Tibia' and 'F' in name:
                y_offset = 0.0#0.05

            if name[:1] == 'L':
                dist = np.array([dist_mm[0],0,-dist_mm[1]])
                y_offset *=-1
            if name[:1] == 'R':
                dist = np.array([-dist_mm[0],0,-dist_mm[1]])

            offset = joints['raw_pos_aligned'][0]-dist
            
            #x_vals = joints['raw_pos_aligned'].transpose()[0] - offset[0]
            #y_vals = joints['raw_pos_aligned'].transpose()[1] - y_offset
            #z_vals = joints['raw_pos_aligned'].transpose()[2] - offset[2]
            x_vals=[]
            y_vals=[]
            z_vals=[]
            for pnt in joints['raw_pos_aligned']:
                x_new = pnt[0] - offset[0]
                y_new = pnt[1] - y_offset
                z_new = pnt[2] - offset[2]

                if scale_procrustes:
                    y_new *= procrustes_factor
                
                x_vals.append(x_new)
                y_vals.append(y_new)
                z_vals.append(z_new)
            
            #if scale_procrustes:
            #    y_vals *= procrustes_factor
                       
            joints['raw_pos_aligned'] = np.array([x_vals,y_vals,z_vals]).transpose()

            if k == 'Coxa':
                joints['fixed_pos_aligned'] = np.mean(joints['raw_pos_aligned'],axis=0)
    
    data_3d = recalculate_lengths(data_3d)
    
    return data_3d
'''
'''
def rescale_using_2d_data2(data_3d,data_2d,cams_info,exp_dir,pixelSize=[5.86e-3,5.86e-3],scale_procrustes = True,procrustes_factor=0.5):
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

    #draw_legs_from_2d(left_view, exp_dir,saveimgs=True)   

    for name, leg in data_3d.items():  
        for k, joints in leg.items():
            if scale_procrustes and k =='Coxa':
                joints['fixed_pos_aligned'][1] = joints['fixed_pos_aligned'][1]*procrustes_factor
                x_3d = joints['raw_pos_aligned'].transpose()[0] 
                y_3d = joints['raw_pos_aligned'].transpose()[1] * procrustes_factor
                z_3d = joints['raw_pos_aligned'].transpose()[2]
                scaled_data = np.array([x_3d,y_3d,z_3d]).transpose()
                joints['raw_pos_aligned'] = scaled_data
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
                #if scale_procrustes:
                #    y_3d = joints['raw_pos_aligned'].transpose()[1]*0.5
                #else:
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
                #print(name,x_factor,y_factor,z_factor)
                for i in range(len(x_3d)):
                    x_3d_scaled.append(x_3d_zero + (x_3d[i] - x_3d_zero) * x_factor)
                    if scale_procrustes:
                        y_3d_scaled.append((y_3d_zero + (y_3d[i] - y_3d_zero) * y_factor)*procrustes_factor)
                    else:
                        y_3d_scaled.append(y_3d_zero + (y_3d[i] - y_3d_zero) * y_factor)
                    z_3d_scaled.append(z_3d_zero + (z_3d[i] - z_3d_zero) * z_factor)
                
                scaled_data = np.array([x_3d_scaled,y_3d_scaled,z_3d_scaled]).transpose()

                joints['raw_pos_aligned'] = scaled_data
    
    data_3d = recalculate_lengths(data_3d)
    
    return data_3d
'''

def recalculate_lengths(data):
    for name, leg in data.items():
        for segment, body_part in leg.items():
            if segment != 'Claw':
                dist = []
                metric = 'raw_pos_aligned'
                if 'Coxa' in segment:
                    next_segment = 'Femur'
                    
                if 'Femur' in segment:
                    next_segment = 'Tibia'
                    
                if 'Tibia' in segment:
                    next_segment = 'Tarsus'
                    
                if 'Tarsus' in segment:
                    next_segment = 'Claw'
                    
                for i, point in enumerate(body_part[metric]):
                    a = point
                    b = data[name][next_segment][metric][i] 
                    dist.append(np.linalg.norm(a-b))
                        
                body_part['mean_length']=np.mean(dist)
    return data

def draw_legs_from_2d(cam_view,exp_dir,begin=0,end=0,saveimgs = False):
    key = [k for k in cam_view.keys() if 'points2d' in k][0]
    data = cam_view[key]
    cam_id = cam_view['cam_id']
    
    colors = {'LF_leg':(0,0,204),'LM_leg':(51,51,255),'LH_leg':(102,102,255),'RF_leg':(153,76,0),'RM_leg':(255,128,0),'RH_leg':(255,178,102)}
    
    if end == 0:
        end = len(data['LF_leg']['Coxa'])
    
    for frame in range(begin,end):
        df3d_dir = exp_dir.find('df3d')
        folder = exp_dir[:df3d_dir]
        img_name = folder + 'camera_' + str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
        img = cv.imread(img_name)
        for leg, body_parts in data.items():
            if leg[0] == 'L' or leg[0]=='R':
                color = colors[leg]
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
            new_folder = 'results/'+file_name.replace('.pkl','/')+'tracking_2d/'
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
