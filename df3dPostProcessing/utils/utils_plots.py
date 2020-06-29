import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

def plot_angles(angles,key,degrees=True):
    leg=angles[key+'_leg']
   
    fig_top = plt.figure()  
    ax_2d = plt.axes()
    for name, angle in leg.items():
        if degrees:
            angle = np.array(angle)*180/np.pi
        ax_2d.plot(angle, label=name)

    ax_2d.legend()
    ax_2d.set_xlabel('time')
    ax_2d.set_ylabel('deg')
    ax_2d.grid(True)
    
    plt.show()

def plot_SG_angles(angles,degrees=True):
    frames, segments = angles.shape
    fig_top = plt.figure()  
    ax_2d = plt.axes()
    labels = ['pitch','yaw','roll','th_fe','th_ti','th_ta']
    for i in range(6):
        #if degrees:
        #    angles[:,i] = np.array(angles[:,1])*180/np.pi
        ax_2d.plot(angles[:,i], label=labels[i])

    ax_2d.legend()
    ax_2d.set_xlabel('time')
    ax_2d.set_ylabel('deg')
    ax_2d.grid(True)
    
    plt.show()

def plot_3d_and_2d(data, plane='xz', begin=0,end=0,metric = 'raw_pos_aligned', savePlot = False):
    colors = {'LF':(1,0,0),'LM':(0,1,0),'LH':(0,0,1),'RF':(1,1,0),'RM':(1,0,1),'RH':(0,1,1)}
    if end == 0:
        end = len(data['LF_leg']['Coxa'][metric])
    for frame in range(begin,end):
        print(frame)
        fig_3d = plt.figure()
        fig_top = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_2d = plt.axes()
        for segment, landmark in data.items():
            if 'L' in segment:
                x=[]
                y=[]
                z=[]
                for key, val in landmark.items():
                    try:
                        x.append(val[metric][frame][0])
                        y.append(val[metric][frame][1])
                        z.append(val[metric][frame][2])
                    except:
                        x.append(val[frame][0])
                        y.append(val[frame][1])
                        z.append(val[frame][2])
                ax_3d.plot(x, y, z, '-o', label=segment, color = colors[segment[:2]])
                if plane == 'xy':
                    ax_2d.plot(x, y, '--x', label=segment)
                if plane == 'xz':
                    ax_2d.plot(x, z, '--x', label=segment)
                if plane == 'yz':
                    ax_2d.plot(y, z, '--x', label=segment)

        ax_3d.legend(loc='upper left')
        ax_3d.set_title('Tracking 3D')
        ax_3d.set_xlabel('X (mm)')
        ax_3d.set_ylabel('Y (mm)')
        ax_3d.set_zlabel('Z (mm)')
        ax_3d.set_xlim(-2.1, 3.5)
        ax_3d.set_ylim(-2.1, 3)
        ax_3d.set_zlim(0.1, 2.3)
        ax_3d.grid(True)

        if savePlot:
            folder = 'results/tracking_3d_leftside/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            figName = folder + 'pos_3d_frame_'+str(frame)+'.png'
            fig_3d.savefig(figName)


        ax_2d.legend()
        ax_2d.set_xlabel('X (mm)')
        ax_2d.set_ylabel('Z (mm)')
        ax_2d.grid(True)

        #plt.show()

def plot_fixed_coxa(aligned_dict):
    metric = 'fixed_pos_aligned'
    fig_3d = plt.figure()
    fig_top = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_2d = plt.axes()
    for leg, joints in aligned_dict.items():
        x=[]
        y=[]
        z=[]
        for joint, data in joints.items():
            if 'Coxa' in joint:
                x.append(data[metric][0])
                y.append(data[metric][1])
                z.append(data[metric][2])
        ax_3d.plot(x, y, z, '-o', label=leg)
        ax_2d.plot(x, y, '--x', label=leg)

    ax_3d.legend()
    ax_3d.set_xlabel('X Label')
    ax_3d.set_ylabel('Y Label')
    ax_3d.set_zlabel('Z Label')
    ax_3d.grid(True)

    ax_2d.legend()
    ax_2d.set_xlabel('X (mm)')
    ax_2d.set_ylabel('Y (mm)')
    ax_2d.grid(True)
    
    plt.show()


def plot_RoM_distance(data_dict, axis, savePlot=False,gen='fly'):
    #names = ['RF_leg','LF_leg','RM_leg','LM_leg','RH_leg','LH_leg'] 
    plt.figure() 
    for name, leg in data_dict.items(): 
        x_amp = [] 
        y_amp = [] 
        z_amp = [] 
        j = list(leg.keys()) 
        for k, joints in leg.items(): 
            #plt.figure()
            try: 
                x_pos = joints['raw_pos_aligned'].transpose()[0] 
                y_pos = joints['raw_pos_aligned'].transpose()[1] 
                z_pos = joints['raw_pos_aligned'].transpose()[2]
            except:
                x_pos = joints.transpose()[0] 
                y_pos = joints.transpose()[1] 
                z_pos = joints.transpose()[2]
            x_amp.append(np.max(x_pos)-np.min(x_pos)) 
            y_amp.append(np.max(y_pos)-np.min(y_pos)) 
            z_amp.append(np.max(z_pos)-np.min(z_pos))         
            #print(name + ': ' + k) 
            #print('X(amp,max,min): ', np.max(x_pos)-np.min(x_pos), np.max(x_pos), np.min(x_pos)) 
            #print('Y(amp,max,min): ', np.max(y_pos)-np.min(y_pos), np.max(y_pos), np.min(y_pos))
            #print('Z(amp,max,min): ', np.max(z_pos)-np.min(z_pos), np.max(z_pos), np.min(z_pos))
        if axis == 'x':
            plt.plot(j,x_amp,'--o',label=name)
        if axis == 'y':
            plt.plot(j,y_amp,'--o',label=name)
        if axis == 'z':
            plt.plot(j,z_amp,'--o',label=name)
     
    plt.title('Range of motion: ' + axis + '-axis') 
    plt.xlabel('Joints') 
    plt.ylabel('Distance (mm)')
    plt.ylim(-0.1, 4.1)
    plt.legend(loc='upper left') 
    plt.grid()
    if savePlot:
        figName = gen + '_rom_distance_'+axis+'_axis.png'
        plt.savefig(figName)
    plt.show()

def plot_pos_series(data_dict, legs, joints, savePlot=False, gen = 'fly'):
    for name in legs: 
        x_amp = [] 
        y_amp = [] 
        z_amp = [] 
        j = list(data_dict[name].keys()) 
        for k, joint in data_dict[name].items():
            if k in joints:
                plt.figure()
                try: 
                    x_pos = joint['raw_pos_aligned'].transpose()[0] 
                    y_pos = joint['raw_pos_aligned'].transpose()[1] 
                    z_pos = joint['raw_pos_aligned'].transpose()[2]
                except:
                    x_pos = joint.transpose()[0] 
                    y_pos = joint.transpose()[1] 
                    z_pos = joint.transpose()[2]
                time = np.arange(len(x_pos))/100
                plt.plot(time,x_pos,label='x') 
                plt.plot(time,y_pos,label='y') 
                plt.plot(time,z_pos,label='z')     
                plt.title(k + ' ' + name + ': 3D position') 
                plt.xlabel('Time (s)') 
                plt.ylabel('Distance (mm)')
                plt.ylim(-3, 4) 
                plt.legend() 
                plt.grid()
                if savePlot:
                    figName = gen+'_' + name+'_'+k+'_pos.png'
                    plt.savefig(figName)
                plt.show()


def plot_legs_from_angles(angles,data_dict,exp_dir,begin=0,end=0,saveimgs = False):

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    if end == 0:
        end = len(leg['yaw'])

    for frame in range(begin, end):
        
        for name, leg in angles.items():
            if 'F_' in name:
                r1 = R.from_euler('zyx',[leg['roll'][frame],leg['pitch'][frame],leg['yaw'][frame]])
                r2 = R.from_euler('zyx',[0,leg['th_fe'][frame],0])
                r3 = R.from_euler('y',leg['th_ti'][frame])
                r4 = R.from_euler('y',leg['th_ta'][frame])
            else:
                r1 = R.from_euler('zyx',[leg['roll'][frame]-np.pi,leg['pitch'][frame],leg['yaw'][frame]])
                r2 = R.from_euler('x',leg['th_fe'][frame])
                r3 = R.from_euler('x',leg['th_ti'][frame])
                r4 = R.from_euler('x',leg['th_ta'][frame])

            coxa_pos = data_dict[name]['Coxa']['fixed_pos_aligned']
            real_pos_femur = data_dict[name]['Femur']['raw_pos_aligned'][frame]
            real_pos_tibia = data_dict[name]['Tibia']['raw_pos_aligned'][frame]
            real_pos_tarsus = data_dict[name]['Tarsus']['raw_pos_aligned'][frame]
            real_pos_claw = data_dict[name]['Claw']['raw_pos_aligned'][frame]
            
            l_coxa = np.linalg.norm(coxa_pos-real_pos_femur)#data_dict[name]['Coxa']['mean_length']*0.08
            l_femur = np.linalg.norm(real_pos_femur-real_pos_tibia)#data_dict[name]['Femur']['mean_length']*0.08
            l_tibia = np.linalg.norm(real_pos_tibia-real_pos_tarsus)#data_dict[name]['Tibia']['mean_length']*0.08
            l_tarsus = np.linalg.norm(real_pos_tarsus-real_pos_claw)#data_dict[name]['Tarsus']['mean_length']*0.08
            
            fe_init_pos = np.array([0,0,-l_coxa])
            ti_init_pos = np.array([0,0,-(l_coxa+l_femur)])
            ta_init_pos = np.array([0,0,-(l_coxa+l_femur+l_tibia)])
            claw_init_pos = np.array([0,0,-(l_coxa+l_femur+l_tibia+l_tarsus)])
            
            femur_pos = r1.apply(fe_init_pos) + coxa_pos
            tibia_pos = r1.apply(r2.apply(ti_init_pos)) + coxa_pos
            
            tarsus_pos = r3.apply(r2.apply(r1.apply(ta_init_pos))) + coxa_pos
            claw_pos = r4.apply(r3.apply(r2.apply(r1.apply(claw_init_pos)))) + coxa_pos

            pos_3d = np.array([coxa_pos,femur_pos,tibia_pos]).transpose()
            x = pos_3d[0]
            y = pos_3d[1]
            z = pos_3d[2]
            ax_3d.plot(x, y, z, '-o', label=name)

            

            real_pos_3d = np.array([coxa_pos,real_pos_femur,real_pos_tibia,real_pos_tarsus,real_pos_claw]).transpose()
            real_x = real_pos_3d[0]
            real_y = real_pos_3d[1]
            real_z = real_pos_3d[2]
            ax_3d.plot(real_x, real_y, real_z, '--x', label=name+'_real')

            
        if saveimgs:
            file_name = exp_dir.split('/')[-1]
            new_folder = 'results/tracking_2d/'+file_name.replace('.pkl','/')
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
                name = new_folder + 'camera_' + str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
    ax_3d.legend()
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.grid(True)
    plt.show()
