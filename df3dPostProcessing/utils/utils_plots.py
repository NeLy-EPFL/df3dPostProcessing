import numpy as np
import matplotlib.pyplot as plt

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

def plot_3d_and_2d(data, metric = 'raw_pos'):
    fig_3d = plt.figure()
    fig_top = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_2d = plt.axes()
    for segment, landmark in data.items():
        x=[]
        y=[]
        z=[]
        for key, val in landmark.items():
            try:
                x.append(val[metric][6][0])
                y.append(val[metric][6][1])
                z.append(val[metric][6][2])
            except:
                x.append(val[0][0])
                y.append(val[0][1])
                z.append(val[0][2])
        ax_3d.plot(x, y, z, '-o', label=segment)
        ax_2d.plot(x, z, '--x', label=segment)

    ax_3d.legend()
    ax_3d.set_xlabel('X Label')
    ax_3d.set_ylabel('Y Label')
    ax_3d.set_zlabel('Z Label')
    ax_3d.grid(True)

    ax_2d.legend()
    ax_2d.set_xlabel('X (mm)')
    ax_2d.set_ylabel('Z (mm)')
    ax_2d.grid(True)
    
    plt.show()

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
    names = ['RF_leg','LF_leg','RM_leg','LM_leg','RH_leg','LH_leg'] 
    plt.figure() 
    for name in names: 
        x_amp = [] 
        y_amp = [] 
        z_amp = [] 
        j = list(data_dict[name].keys()) 
        for k, joints in data_dict[name].items(): 
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
            #plt.plot(x_pos,label='x') 
            #plt.plot(y_pos,label='y') 
            #plt.plot(z_pos,label='z') 
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
    plt.ylim(0, 4.1)
    plt.legend() 
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
