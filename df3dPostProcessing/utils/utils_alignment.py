import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def align_data(exp_dict,rescale):
    fix = fixed_lengths_and_base_point(exp_dict)
    align = align_model(fix)
    if rescale:
        align = rescale_using_2d_data(align)

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


def rescale_using_2d_data(exp_dict):
    """
    Re-scale 3d data using 2d data
    """
    print('Re-scale 3d data using 2d data')
    return exp_dict
