import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error
from sklearn import svm
from . import utils_angles
import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats as stests
from .utils_angles import calculate_forward_kinematics
from ikpy.chain import Chain 
from ikpy.link import OriginLink, URDFLink


def plot_angles(angles,key,degrees=True):
    leg=angles[key+'_leg']
   
    fig_top = plt.figure()  
    ax_2d = plt.axes()
    for name, angle in leg.items():
        if degrees:
            angle = np.array(angle)*180/np.pi
        ax_2d.plot(angle, label=name)

    title = key+' leg angles'
    ax_2d.legend()
    plt.title(title)
    ax_2d.set_xlabel('time')
    ax_2d.set_ylabel('deg')
    ax_2d.set_ylim(-230,200)
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
            if 'L' in segment or 'R' in segment:
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


def plot_legs_from_angles(angles,data_dict,exp_dir,begin=0,end=0,plane='xz',saveImgs = False, dir_name='km', roll_tr_angles = {}, ik_angles = False, pause = False):

    #colors_real= {'LF_leg':(1,0,0),'LM_leg':(0,1,0),'LH_leg':(0,0,1),'RF_leg':(1,1,0),'RM_leg':(1,0,1),'RH_leg':(0,1,1)}
    #colors = {'LF_leg':(1,0.5,0.5),'LM_leg':(0.5,1,0.5),'LH_leg':(0.5,0.5,1),'RF_leg':(1,1,0.5),'RM_leg':(1,0.5,1),'RH_leg':(0.5,1,1)}

    colors = {'LF_leg':(204/255,0,0),'LM_leg':(1,51/255,51/255),'LH_leg':(1,102/255,102/255),'RF_leg':(0,76/255,153/255),'RM_leg':(0,0.5,1),'RH_leg':(102/255,178/255,1)}

    fig_3d = plt.figure()
    view_2d = plt.figure()
    
    if end == 0:
        end = len(angles['LF_leg']['yaw'])

    for frame in range(begin, end):
        print('\rFrame: '+str(frame),end='')
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_2d = plt.axes()

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_xlim(-3, 4)
        ax_3d.set_ylim(-3, 3)
        ax_3d.set_zlim(-0.5, 4)
        ax_3d.grid(True)

        ax_2d.set_xlabel(plane[0])
        ax_2d.set_ylabel(plane[1])
        ax_2d.grid(True)

        for name, leg in angles.items():
            coxa_pos = data_dict[name]['Coxa']['fixed_pos_aligned']
            real_pos_femur = data_dict[name]['Femur']['raw_pos_aligned'][frame]
            real_pos_tibia = data_dict[name]['Tibia']['raw_pos_aligned'][frame]
            real_pos_tarsus = data_dict[name]['Tarsus']['raw_pos_aligned'][frame]
            real_pos_claw = data_dict[name]['Claw']['raw_pos_aligned'][frame]

            if roll_tr_angles:
                roll_tr_best = roll_tr_angles[name]['roll_tr'][frame]
            else:
                roll_tr_best = 0

            if ik_angles:
                pos_3d = fk_from_ik(leg, name, data_dict, frame).transpose()
            else:
                pos_3d = utils_angles.calculate_forward_kinematics(name, frame, leg, data_dict, extraDOF={'roll_tr':roll_tr_best},ik_angles=ik_angles).transpose()

                      
            x = pos_3d[0]
            y = pos_3d[1]
            z = pos_3d[2]
            ax_3d.plot(x, y, z, '-o', label=name, color = colors[name])            

            real_pos_3d = np.array([coxa_pos,real_pos_femur,real_pos_tibia,real_pos_tarsus,real_pos_claw]).transpose()
            real_x = real_pos_3d[0]
            real_y = real_pos_3d[1]
            real_z = real_pos_3d[2]
            ax_3d.plot(real_x, real_y, real_z, '--x', label=name+'_real', color = colors[name])

            if plane == 'xy':
                ax_2d.set_xlim(-3, 4)
                ax_2d.set_ylim(-3, 3)
                ax_2d.plot(real_x, real_y, '--x', label=name+'_real', color = colors[name])
                ax_2d.plot(x, y, '-o', label=name, color = colors[name])
            if plane == 'xz':
                ax_2d.set_xlim(-3, 4)
                ax_2d.set_ylim(-0.5, 4)
                ax_2d.plot(real_x, real_z, '--x', label=name+'_real', color = colors[name])
                ax_2d.plot(x, z, '-o', label=name, color = colors[name])
            if plane == 'yz':
                ax_2d.set_xlim(-3, 3)
                ax_2d.set_ylim(-0.5, 4)
                ax_2d.plot(real_y, real_z, '--x', label=name+'_real', color = colors[name])
                ax_2d.plot(y, z, '-o', label=name, color = colors[name])
            
        if saveImgs:
            file_name = exp_dir.split('/')[-1]
            new_folder_3d = 'results/'+file_name.replace('.pkl','/')+dir_name+'_3d/'
            new_folder_2d = 'results/'+file_name.replace('.pkl','/')+dir_name+'_'+plane+'/'
            if not os.path.exists(new_folder_3d):
                os.makedirs(new_folder_3d)
            if not os.path.exists(new_folder_2d):
                os.makedirs(new_folder_2d)
            name_3d = new_folder_3d + 'km_3d' + '_img_' + '{:06}'.format(frame) + '.jpg'
            name_2d = new_folder_2d + 'km_' + plane + '_img_' + '{:06}'.format(frame) + '.jpg'
            fig_3d.savefig(name_3d)
            view_2d.savefig(name_2d)

        if not saveImgs:
            ax_3d.legend()
            view_2d.legend()
            if pause:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(0.001)

        fig_3d.clf()
        view_2d.clf()

    plt.close()


def plot_correlation_matrix(angles,roll_tr,align):
    for name, leg in angles.items():
        claw_h = align[name]['Claw']['raw_pos_aligned'].transpose()[2]
        tarsus_h = align[name]['Tarsus']['raw_pos_aligned'].transpose()[2]
        tibia_h = align[name]['Tibia']['raw_pos_aligned'].transpose()[2]
        
        r = np.corrcoef([roll_tr[name]['best_roll'], claw_h, tarsus_h, tibia_h, leg['yaw'], leg['pitch'], leg['roll'], leg['th_fe'], leg['th_ti'], leg['th_ta']])

        labels = ['roll_tr']+list(leg.keys())
        
        plot_heatmap(r,labels,name)

def plot_heatmap(corr, labels,title):

    ax = sns.heatmap( 
        corr,  
        #vmin=-1, vmax=1, center=0, 
        annot=True, 
        xticklabels = labels,
        yticklabels = labels, 
        #cmap = 'seismic' 
        )
    plt.title(title)
    plt.show()

def calculate_svm_from_angles(angles, diff,align):
  
    #angles=np.load('angles.pkl',allow_pickle=True)  
    #diff=np.load('diff_general.pkl',allow_pickle=True)  
        
    #name = 'LF_leg' 
    #name2 = 'RF_leg' 
    mat=np.zeros((len(angles.keys()),len(angles.keys()))) 
    
    for i, (name, leg) in enumerate(angles.items()):   
        claw_h = align[name]['Claw']['raw_pos_aligned'].transpose()[2]  
        tarsus_h = align[name]['Tarsus']['raw_pos_aligned'].transpose()[2] 
    
        tibia_h = align[name]['Tibia']['raw_pos_aligned'].transpose()[2]  
        yaw = angles[name]['yaw'] 
        pitch = angles[name]['pitch'] 
        roll = angles[name]['roll'] 
        th_fe = angles[name]['th_fe'] 
        th_ti = angles[name]['th_ti'] 
        th_ta = angles[name]['th_ta'] 
        X = np.array([claw_h, tarsus_h, tibia_h, roll, th_fe]) 
        y = np.array(diff[name]['best_roll']) 
        clf = svm.SVR() 
        clf.fit(X.T,y) 
     
        for j, (name2, leg) in enumerate(angles.items()):  
        
            claw_h = align[name2]['Claw']['raw_pos_aligned'].transpose()[2]
      
            tarsus_h = align[name2]['Tarsus']['raw_pos_aligned'].transpose()[2]
            tibia_h = align[name2]['Tibia']['raw_pos_aligned'].transpose()[2]   
            yaw = angles[name2]['yaw']  
            pitch = angles[name2]['pitch']  
            roll = angles[name2]['roll']  
            th_fe = angles[name2]['th_fe']  
            th_ti = angles[name2]['th_ti']  
            th_ta = angles[name2]['th_ta']  
            X2 = np.array([claw_h, tarsus_h, tibia_h, roll, th_fe])  
            y_real = np.array(diff[name2]['best_roll']) 
            
            y_pred = clf.predict(X2.T) 
            mse = mean_squared_error(y_real,y_pred) 
            rmse = np.sqrt(mse)*180/np.pi 
            mat[i][j] = rmse 
            print(name + ' to ' + name2, rmse) 
        print()

    plot_heatmap(mat,list(angles.keys()),'RMSE')

def plot_error(errors_dict):
    legs = list(errors_dict.keys())
    angles = list(errors_dict['LF_leg'].keys())
    df_errors = pd.DataFrame()

    colors = [(204/255,0,0),(1,51/255,51/255),(1,102/255,102/255),(0,76/255,153/255),(0,0.5,1),(102/255,178/255,1)]

    for leg in legs:
        for angle in angles:
            vals = []
            for err in errors_dict[leg][angle]['min_error']:
                vals.append(err[0])

            df_vals = pd.DataFrame(vals,columns=['error (mm)'])
            df_vals['leg'] = leg
            df_vals['angle'] = angle

            df_errors = df_errors.append(df_vals, ignore_index = True)
    
    for angle1 in angles:
        x1 = df_errors['error (mm)'].loc[df_errors['angle']==angle1]
        print(angle1 + ' mean/std = ' + str(np.mean(x1)) + ' /+- ' + str(np.std(x1)))
        for angle2 in angles[angles.index(angle1)+1:]:
        #if angle != 'base':            
            x2 = df_errors['error (mm)'].loc[df_errors['angle']==angle2]
            ztest , pval = stests.ztest(x1, x2=x2, value=0, alternative='two-sided')

            print(angle1 + ' vs ' + angle2 + ': ', ztest, pval,)
            if pval > 0.001:
                print(angle1 + " is not statistically different from " + angle2)
        print()
    
    ax = sns.violinplot(x='angle', y='error (mm)', data=df_errors, color="0.8")
    for violin, alpha in zip(ax.collections[::2], [0.7]*len(angles)):
        violin.set_alpha(alpha)
    ax = sns.stripplot(x='angle', y='error (mm)', hue='leg', data=df_errors, jitter=True, zorder=0, palette = colors,size=3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.title('Comparison adding an extra DOF')
    plt.show()
    
    return df_errors

def calculate_inverse_kinematics(data_dict, init_angles=[],roll_tr=False):
    angles_ik = {}
    
    for name, leg in data_dict.items():
        if roll_tr:
            angles_ik[name]={'roll':[],'pitch':[],'yaw':[],'th_fe':[],'roll_tr':[],'th_ti':[],'th_ta':[]}
        else:
            angles_ik[name]={'roll':[],'pitch':[],'yaw':[],'th_fe':[],'th_ti':[],'th_ta':[]}
        l_coxa = leg['Coxa']['mean_length']
        l_femur = leg['Femur']['mean_length']
        l_tibia = leg['Tibia']['mean_length']
        l_tarsus = leg['Tarsus']['mean_length']
        coxa_pos = leg['Coxa']['fixed_pos_aligned']

        leg_chain = create_kinematic_chain(name,l_coxa,l_femur,l_tibia,l_tarsus,roll_tr)

        if not init_angles:
            init_pos = [0]*len(leg_chain.links)
        else:
            init_pos = [0]
            for key in angles_ik[name].keys():
                if 'roll' in key and '_tr' not in key:
                    if 'LM' in name or 'LH' in name:
                        init_pos.append(-(np.pi/2 + init_angles[name][key][0]))
                    elif 'RM' in name or 'RH' in name:
                        init_pos.append((np.pi/2 + init_angles[name][key][0]))
                    else:
                        init_pos.append(init_angles[name][key][0])
                else:
                    init_pos.append(init_angles[name][key][0])
            init_pos.append(0)

        for i, pos in enumerate(leg['Claw']['raw_pos_aligned']):
            print('\r'+name+' frame: '+str(i),end='')
            pos_norm = pos - coxa_pos
            angles = leg_chain.inverse_kinematics(pos_norm,initial_position=init_pos)
            angles_ik[name]['roll'].append(angles[1])
            angles_ik[name]['pitch'].append(angles[2])
            angles_ik[name]['yaw'].append(angles[3])
            angles_ik[name]['th_fe'].append(angles[4])
            if roll_tr:
                angles_ik[name]['roll_tr'].append(angles[5])
                angles_ik[name]['th_ti'].append(angles[6])
                angles_ik[name]['th_ta'].append(angles[7])
            else:
                angles_ik[name]['th_ti'].append(angles[5])
                angles_ik[name]['th_ta'].append(angles[6])

    return angles_ik

def create_kinematic_chain(name,l_co,l_fe,l_ti,l_ta,roll_tr = False):
    if roll_tr:
        leg_chain = Chain(name=name, links=[ 
        OriginLink(), 
        URDFLink( 
          name="CoxaRoll", 
          translation_vector=[0, 0, 0], 
          orientation=[0, 0, 0], 
          rotation=[0, 0, 1], 
        ), 
        URDFLink( 
          name="CoxaPitch", 
          translation_vector=[0, 0, 0], 
          orientation=[0, 0, 0], 
          rotation=[0, 1, 0], 
        ), 
        URDFLink(  
          name="CoxaYaw",  
          translation_vector=[0, 0, 0],  
          orientation=[0, 0, 0],  
          rotation=[1, 0, 0],  
        ), 
        URDFLink(  
          name="th_fe",  
          translation_vector=[0, 0, -l_co],  
          orientation=[0, 0, 0],  
          rotation=[0, 1, 0],  
        ),
        URDFLink(  
          name="roll_tr",  
          translation_vector=[0, 0, 0],  
          orientation=[0, 0, 0],  
          rotation=[1, 0, 0],  
        ),
        URDFLink(  
          name="th_ti",  
          translation_vector=[0, 0, -l_fe],  
          orientation=[0, 0, 0],  
          rotation=[0, 1, 0],  
        ), 
        URDFLink( 
          name="th_ta", 
          translation_vector=[0, 0, -l_ti], 
          orientation=[0, 0, 0], 
          rotation=[0, 1, 0], 
        ), 
        URDFLink(  
          name="Claw",  
          translation_vector=[0, 0, -l_ta],  
          orientation=[0, 0, 0],  
          rotation=[0, 0, 0],  
        ), 
        ])
    else:
        leg_chain = Chain(name=name, links=[ 
        OriginLink(), 
        URDFLink( 
          name="CoxaRoll", 
          translation_vector=[0, 0, 0], 
          orientation=[0, 0, 0], 
          rotation=[0, 0, 1], 
        ), 
        URDFLink( 
          name="CoxaPitch", 
          translation_vector=[0, 0, 0], 
          orientation=[0, 0, 0], 
          rotation=[0, 1, 0], 
        ), 
        URDFLink(  
          name="CoxaYaw",  
          translation_vector=[0, 0, 0],  
          orientation=[0, 0, 0],  
          rotation=[1, 0, 0],  
        ), 
        URDFLink(  
          name="th_fe",  
          translation_vector=[0, 0, -l_co],  
          orientation=[0, 0, 0],  
          rotation=[0, 1, 0],  
        ),
        URDFLink(  
          name="th_ti",  
          translation_vector=[0, 0, -l_fe],  
          orientation=[0, 0, 0],  
          rotation=[0, 1, 0],  
        ), 
        URDFLink( 
          name="th_ta", 
          translation_vector=[0, 0, -l_ti], 
          orientation=[0, 0, 0], 
          rotation=[0, 1, 0], 
        ), 
        URDFLink(  
          name="Claw",  
          translation_vector=[0, 0, -l_ta],  
          orientation=[0, 0, 0],  
          rotation=[0, 0, 0],  
        ), 
        ])

    return leg_chain

def fk_from_ik(leg, name, data_dict, frame, roll_tr = False):

    l_coxa = data_dict[name]['Coxa']['mean_length']
    l_femur = data_dict[name]['Femur']['mean_length']
    l_tibia = data_dict[name]['Tibia']['mean_length']
    l_tarsus = data_dict[name]['Tarsus']['mean_length']

    leg_chain = create_kinematic_chain(name,l_coxa,l_femur,l_tibia,l_tarsus,roll_tr)
    
    if roll_tr:
        vect = [0,leg['roll'][frame],leg['pitch'][frame],leg['yaw'][frame],leg['th_fe'][frame],leg['roll_tr'][frame],leg['th_ti'][frame],leg['th_ta'][frame],0]
    else:
        vect = [0,leg['roll'][frame],leg['pitch'][frame],leg['yaw'][frame],leg['th_fe'][frame],leg['th_ti'][frame],leg['th_ta'][frame],0]

    pred = leg_chain.forward_kinematics(vect,full_kinematics=True)

    fe_init_pos = np.array([0,0,-l_coxa,1])
    ti_init_pos = np.array([0,0,-l_femur,1])
    ta_init_pos = np.array([0,0,-l_tibia,1])
    claw_init_pos = np.array([0,0,-l_tarsus,1])

    real_pos_coxa = data_dict[name]['Coxa']['fixed_pos_aligned']
    pred_coxa =  pred[3].transpose()[3][:3] + real_pos_coxa
    pred_femur =  pred[4].transpose()[3][:3] + real_pos_coxa
    pred_tibia = pred[5].transpose()[3][:3] + real_pos_coxa
    pred_tarsus = pred[6].transpose()[3][:3] + real_pos_coxa
    pred_claw = pred[7].transpose()[3][:3] + real_pos_coxa
    
    pred_pos = np.array([pred_coxa,pred_femur,pred_tibia,pred_tarsus,pred_claw])

    return pred_pos


def calculate_min_error(angles,data_dict,begin=0,end=0,extraDOF = ['base'],legs_to_check=[]):
    #extraKeys = ['roll_tr','yaw_tr','roll_ti','yaw_ti','roll_ta','yaw_ta']

    errors_dict = {}
    
    if end == 0:
        end = len(angles['LF_leg']['yaw'])

    for frame in range(begin, end):
        print('\rFrame: '+str(frame),end='')

        for name, leg in angles.items():

            if legs_to_check:
                if not name in legs_to_check:
                    break
                
            if not name in errors_dict.keys():
                errors_dict[name] = dict.fromkeys(extraDOF)

            for key in extraDOF:
                if not errors_dict[name][key]:
                    errors_dict[name][key] = {'min_error':[],'best_angle':[]}
                
                #coxa_pos = data_dict[name]['Coxa']['fixed_pos_aligned']
                real_pos_femur = data_dict[name]['Femur']['raw_pos_aligned'][frame]
                real_pos_tibia = data_dict[name]['Tibia']['raw_pos_aligned'][frame]
                real_pos_tarsus = data_dict[name]['Tarsus']['raw_pos_aligned'][frame]
                real_pos_claw = data_dict[name]['Claw']['raw_pos_aligned'][frame]
            
                min_error = [100000000,0,0,0,0]
                best_angle = 0
                if key == 'base':
                    pos_3d = calculate_forward_kinematics(name, frame, leg, data_dict)
                    #pos_3d = calculate_forward_kinematics(name, frame, leg, data_dict,extraDOF={'roll_tr':angles[name]['roll_tr'][frame]})

                    d_fe = np.linalg.norm(pos_3d[1]-real_pos_femur)
                    d_ti = np.linalg.norm(pos_3d[2]-real_pos_tibia) 
                    d_ta = np.linalg.norm(pos_3d[3]-real_pos_tarsus)
                    d_claw = np.linalg.norm(pos_3d[4]-real_pos_claw)

                    d_tot = d_fe + d_ti + d_ta + d_claw
                           
                    errors_dict[name][key]['min_error'].append([d_tot,d_fe,d_ti,d_ta,d_claw])
                elif key == 'base_rollTr':
                    #pos_3d = calculate_forward_kinematics(name, frame, leg, data_dict)
                    pos_3d = calculate_forward_kinematics(name, frame, leg, data_dict,extraDOF={'roll_tr':angles[name]['roll_tr'][frame]})

                    d_fe = np.linalg.norm(pos_3d[1]-real_pos_femur)
                    d_ti = np.linalg.norm(pos_3d[2]-real_pos_tibia) 
                    d_ta = np.linalg.norm(pos_3d[3]-real_pos_tarsus)
                    d_claw = np.linalg.norm(pos_3d[4]-real_pos_claw)

                    d_tot = d_fe + d_ti + d_ta + d_claw
                           
                    errors_dict[name][key]['min_error'].append([d_tot,d_fe,d_ti,d_ta,d_claw])
                elif key == 'IK':
                    pos_3d = fk_from_ik(leg, name, data_dict, frame)

                    d_fe = np.linalg.norm(pos_3d[1]-real_pos_femur)
                    d_ti = np.linalg.norm(pos_3d[2]-real_pos_tibia) 
                    d_ta = np.linalg.norm(pos_3d[3]-real_pos_tarsus)
                    d_claw = np.linalg.norm(pos_3d[4]-real_pos_claw)

                    d_tot = d_fe + d_ti + d_ta + d_claw
                           
                    errors_dict[name][key]['min_error'].append([d_tot,d_fe,d_ti,d_ta,d_claw])
                else:
                    for i in range(-180, 180):
                        extra_angle = np.deg2rad(i/2)
                        extra_dict = {key:extra_angle}

                        pos_3d = calculate_forward_kinematics(name, frame, leg, data_dict,extraDOF=extra_dict)

                        d_fe = np.linalg.norm(pos_3d[1]-real_pos_femur)
                        d_ti = np.linalg.norm(pos_3d[2]-real_pos_tibia) 
                        d_ta = np.linalg.norm(pos_3d[3]-real_pos_tarsus)
                        d_claw = np.linalg.norm(pos_3d[4]-real_pos_claw)

                        d_tot = d_fe + d_ti + d_ta + d_claw

                        if d_tot<min_error[0]:
                            min_error = [d_tot,d_fe,d_ti,d_ta,d_claw]
                            best_angle = extra_angle

                    #print(frame,name,key)
                    errors_dict[name][key]['min_error'].append(min_error)
                    errors_dict[name][key]['best_angle'].append(best_angle)
            
    return errors_dict


'''
#####Reorder dictionary
new_errors = {}                                       

extraKeys = ['base','IK','roll_tr','yaw_tr','roll_ti','yaw_ti','roll_ta','yaw_ta'] 
for name, leg in errors.items(): 
    new_errors[name]={} 
    for key in extraKeys: 
        new_errors[name][key] = leg[key]
'''
