import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
from matplotlib.legend_handler import HandlerTuple
from scipy import ndimage
import math
import pickle
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import cv2 as cv
from matplotlib.markers import MarkerStyle

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

def plot_angles_torques_grf(leg_key, angles={}, sim_data='walking', exp_dir='experiment', plot_angles=True, plot_torques=True, plot_grf=True, plot_collisions=True, collisions_across=True, begin=0.0, end=0.0, save_imgs=False, dir_name='dataPlots',torqueScalingFactor=100, grfScalingFactor=10):    

    data2plot={}

    equivalence = {'ThC yaw':['yaw','Coxa_yaw'],
                   'ThC pitch':['pitch','Coxa'],
                   'ThC roll':['roll','Coxa_roll'],
                   'CTr pitch':['th_fe','Femur'],
                   'CTr roll':['roll_tr','Femur_roll'],
                   'FTi pitch':['th_ti','Tibia'],
                   'TiTa pitch':['th_ta','Tarsus1']}
    
    sim_res_folder = '/home/nely/SimFly/kinematic_matching/results/'
    torques_data = sim_res_folder + 'torques_data_ball_' + sim_data + '.pkl'
    if sim_data == 'walking':
        grf_data = sim_res_folder + 'grf_data_ball_' + sim_data + '.pkl'
    elif sim_data == 'grooming':
        collisions_data = sim_res_folder + 'selfCollisions_data_ball_' + sim_data + '.pkl'

    if end == 0:
        end = 9.0

    start = int(begin*100)
    stop = int(end*100)

    if plot_angles:
        angles_raw = angles[leg_key+'_leg']
        data2plot['angles']={}
        for label, match_labels in equivalence.items():
            for key in angles_raw.keys():
                if key in match_labels:
                    data2plot['angles'][label] = angles_raw[key]

    if plot_torques:
        with open(torques_data, 'rb') as fp:
            torques_all = pickle.load(fp)
        torques_raw = {}
        for joint, torque in torques_all.items():
            if leg_key in joint and not 'Haltere' in joint:
                if 'Tarsus' not in joint or 'Tarsus1' in joint:
                    joint_data = joint.split('joint_')
                    label = joint_data[1][2:]
                    torques_raw[label] = torque
        data2plot['torques'] = {}
        for label, match_labels in equivalence.items():
            for key in torques_raw.keys():
                if key in match_labels:
                    data2plot['torques'][label] = torques_raw[key]
    if sim_data == 'walking':
        if plot_grf:
            with open(grf_data, 'rb') as fp:
                data2plot['grf'] = pickle.load(fp)
            grf = []            
            for leg, force in data2plot['grf'].items():
                if leg[:2] == leg_key:
                    grf.append(force.transpose()[0])
            sum_force = np.sum(np.array(grf), axis = 0)
            leg_force = np.delete(sum_force,0)


            #w_size = int(math.ceil(len(leg_force)/9*0.025))
            #filtered_force = ndimage.median_filter(leg_force,size=w_size)
    if sim_data == 'grooming':
        if plot_collisions:
            with open(collisions_data, 'rb') as fp:
                data2plot['collisions'] = pickle.load(fp)
            collisions=[]
            contra_coll = []
            ant_coll = []
            for segment, coll in data2plot['collisions'].items():
                for key, forces in coll.items():
                    side = leg_key[0]
                    if side == 'L':
                        contra_lateral_leg = leg_key.replace(side,'R')
                    else:
                        contra_lateral_leg = leg_key.replace(side,'L')
                    antenna = side+'Antenna'
                    if segment[:2] == leg_key:
                        collisions.append([np.linalg.norm(force) for force in forces])
                    elif segment[:2] == contra_lateral_leg:
                        contra_coll.append([np.linalg.norm(force) for force in forces])
                    elif segment == antenna:
                        ant_coll.append([np.linalg.norm(force) for force in forces])
            sum_leg = np.sum(np.array(collisions), axis = 0)
            sum_contra_leg = np.sum(np.array(contra_coll), axis = 0)
            sum_ant = np.sum(np.array(ant_coll), axis = 0)
            leg_force = np.delete(sum_leg,0)
            contra_force = np.delete(sum_contra_leg,0)
            antenna_force = np.delete(sum_ant,0)
            leg_vs_leg = []
            leg_vs_ant = []
            for i, f in enumerate(leg_force):
                if f > 0 and antenna_force[i]>0:
                    leg_vs_ant.append(f)
                    leg_vs_leg.append(0)
                elif f > 0 and contra_force[i]>0:
                    leg_vs_ant.append(0)
                    leg_vs_leg.append(f)                    
                elif f==0:
                    leg_vs_leg.append(0)
                    leg_vs_ant.append(0)
            
    if collisions_across:
        if not plot_grf and sim_data=='walking':
            with open(grf_data, 'rb') as fp:
                grf_val = pickle.load(fp)
            grf = []            
            for leg, force in grf_val.items():
                if leg[:2] == leg_key:
                    grf.append(force.transpose()[0])
            sum_force = np.sum(np.array(grf), axis = 0)
            leg_force = np.delete(sum_force,0)
        if not plot_collisions and sim_data =='grooming':
            with open(collisions_data, 'rb') as fp:
                data2plot['collisions'] = pickle.load(fp)
            collisions=[]
            contra_coll = []
            ant_coll = []
            for segment, coll in data2plot['collisions'].items():
                for key, forces in coll.items():
                    side = leg_key[0]
                    if side == 'L':
                        contra_lateral_leg = leg_key.replace(side,'R')
                    else:
                        contra_lateral_leg = leg_key.replace(side,'L')
                    antenna = side+'Antenna'
                    if segment[:2] == leg_key:
                        collisions.append([np.linalg.norm(force) for force in forces])
                    elif segment[:2] == contra_lateral_leg:
                        contra_coll.append([np.linalg.norm(force) for force in forces])
                    elif segment == antenna:
                        ant_coll.append([np.linalg.norm(force) for force in forces])
            sum_leg = np.sum(np.array(collisions), axis = 0)
            sum_contra_leg = np.sum(np.array(contra_coll), axis = 0)
            sum_ant = np.sum(np.array(ant_coll), axis = 0)
            leg_force = np.delete(sum_leg,0)
            contra_force = np.delete(sum_contra_leg,0)
            antenna_force = np.delete(sum_ant,0)
            leg_vs_leg = []
            leg_vs_ant = []
            for i, f in enumerate(leg_force):
                if f != 0 and antenna_force[i]>0:
                    leg_vs_ant.append(f)
                    leg_vs_leg.append(0)
                elif f != 0 and contra_force[i]>0:
                    leg_vs_ant.append(0)
                    leg_vs_leg.append(f)                    
                elif f==0:
                    leg_vs_leg.append(0)
                    leg_vs_ant.append(0)
                    
        stance_ind = np.where(leg_force>0)[0]
        if stance_ind.size!=0:
            stance_diff = np.diff(stance_ind)
            stance_lim = np.where(stance_diff>1)[0]
            stance=[stance_ind[0]-1]
            for ind in stance_lim:
                stance.append(stance_ind[ind]+1)
                stance.append(stance_ind[ind+1]-1)
            stance.append(stance_ind[-1])
            start_gait_list = np.where(np.array(stance) >= start)[0]
            if len(start_gait_list)>0:
                start_gait = start_gait_list[0]
            else:
                start_gait = start            
            stop_gait_list = np.where(np.array(stance) <= stop)[0]
            if len(stop_gait_list)>0:
                stop_gait = stop_gait_list[-1]+1
            else:
                stop_gait = start_gait
            stance_plot = stance[start_gait:stop_gait]
            if start_gait%2 != 0:
                stance_plot.insert(0,start)
            if len(stance_plot)%2 != 0:
                stance_plot.append(stop)
    
    fig, axs = plt.subplots(len(data2plot.keys()), sharex=True)
    fig.suptitle(dir_name+' '+leg_key+ ' leg')

    for i, (plot, data) in enumerate(data2plot.items()):
        if plot == 'angles':
            for name, angle_rad in data.items():
                time = np.arange(0,len(angle_rad),1)/100
                angle = np.array(angle_rad)*180/np.pi
                axs[i].plot(time[start:stop], angle[start:stop], label=name)
            axs[i].set_ylabel('Joint angle (deg)')

        if plot == 'torques':
            for joint, torque in data.items():
                torque_adj = np.delete(torque,0)
                time = np.arange(0,len(torque_adj),1)/100
                axs[i].plot(time[start:stop], torque_adj[start:stop]*torqueScalingFactor,label=joint)
            axs[i].set_ylabel('Joint torque ' + r'$(\mu Nm)$')

        if plot == 'grf':
            time = np.arange(0,len(leg_force),1)/100
            axs[i].plot(time[start:stop],leg_force[start:stop]*grfScalingFactor,color='black')
            axs[i].set_ylabel('Ball reaction force (mN)')

        if plot == 'collisions':
            time = np.arange(0,len(leg_force),1)/100
            axs[i].plot(time[start:stop],np.array(leg_vs_leg[start:stop])*grfScalingFactor,color='black',label=['Leg vs leg force'])
            axs[i].plot(time[start:stop],np.array(leg_vs_ant[start:stop])*grfScalingFactor,color='dimgray',label=['Leg vs antenna force'])
            axs[i].set_ylabel('Contact forces (mN)')
            
        axs[i].grid(True)
        if plot != 'grf' and i == 0:
            plot_handles, plot_labels = axs[i].get_legend_handles_labels()
            if collisions_across and sim_data=='walking':
                gray_patch = mpatches.Patch(color='gray')
                all_handles = plot_handles + [gray_patch]
                all_labels = plot_labels+ ['Stance']
            elif collisions_across and sim_data=='grooming':
                gray_patch = mpatches.Patch(color='dimgray')
                darkgray_patch = mpatches.Patch(color='darkgray')
                all_handles = plot_handles + [gray_patch] + [darkgray_patch]
                all_labels = plot_labels + ['Foreleg grooming'] + ['Antennal grooming']
            else:
                all_handles = plot_handles
                all_labels = plot_labels
            axs[i].legend(all_handles, all_labels,loc= 'upper right',bbox_to_anchor=(1.135, 1))

        if collisions_across:
            for ind in range(0,len(stance_plot),2):
                if sim_data=='walking':
                    c = 'gray'
                if sim_data=='grooming':
                    if np.sum(leg_vs_leg[stance_plot[ind]:stance_plot[ind+1]])>0:
                        c = 'dimgray'
                    elif np.sum(leg_vs_ant[stance_plot[ind]:stance_plot[ind+1]])>0:
                        c = 'darkgray'
                axs[i].fill_between(time[stance_plot[ind]:stance_plot[ind+1]], 0, 1, facecolor=c, alpha=0.5, transform=axs[i].get_xaxis_transform())
                
        #axs[i].fill_between(time[start:stop], 0, 1, where=leg_force[start:stop] > 0, facecolor='gray', alpha=0.5, transform=axs[i].get_xaxis_transform(),step='pre')
        
    axs[len(axs)-1].set_xlabel('Time (s)')
    plt.show()

'''
def plot_gait_diagram(begin=0,end=0):
    data={}
    sim_res_folder = '/home/nely/SimFly/kinematic_matching/results/'
    grf_data = sim_res_folder + 'grf_data_ball_walking.pkl'

    if end == 0:
        end = 9.0

    start = int(begin*100)
    stop = int(end*100)
    
    with open(grf_data, 'rb') as fp:
        data = pickle.load(fp)
    grf = {'LF':[],'LM':[],'LH':[],'RF':[],'RM':[],'RH':[]}            
    for leg, force in data.items():
        grf[leg[:2]].append(force)

    
    title_plot = 'Gait diagram'
    
    fig, axs = plt.subplots(len(grf.keys()), sharex=True,gridspec_kw={'hspace': 0})
    fig.suptitle(title_plot)

    for i, (leg, force) in enumerate(grf.items()):
        sum_force = np.sum(np.array(force), axis = 0)
        leg_force = np.delete(sum_force,0)
        time = np.arange(0,len(leg_force),1)/100
        stance_ind = np.where(leg_force>0)[0]
        stance_diff = np.diff(stance_ind)
        stance_lim = np.where(stance_diff>1)[0]
        stance=[stance_ind[0]-1]
        for ind in stance_lim:
            stance.append(stance_ind[ind]+1)
            stance.append(stance_ind[ind+1]-1)
        stance.append(stance_ind[-1])
        
        start_gait = np.where(np.array(stance) >= start)[0][0]
        stop_gait = np.where(np.array(stance) <= stop)[0][-1]+1
        stance_plot = stance[start_gait:stop_gait]
        if start_gait%2 != 0:
            stance_plot.insert(0,start)
        if len(stance_plot)%2 != 0:
            stance_plot.append(stop)

        for ind in range(0,len(stance_plot),2):
            axs[i].fill_between(time[stance_plot[ind]:stance_plot[ind+1]], 0, 1, facecolor='black', alpha=1, transform=axs[i].get_xaxis_transform())

        axs[i].set_yticks((0.5,))
        axs[i].set_yticklabels((leg,))

    axs[len(axs)-1].set_xlabel('Time (s)')
    black_patch = mpatches.Patch(color='black', label='Stance')
    axs[0].legend(handles=[black_patch],loc= 'upper right',bbox_to_anchor=(1.1, 1))
    plt.show()
'''

def plot_collisions_diagram(sim_data, begin=0, end=0):
    data={}
    sim_res_folder = '/home/nely/SimFly/kinematic_matching/results/'

    if sim_data == 'walking':
        collisions_data = sim_res_folder + 'grf_data_ball_walking.pkl'
    elif sim_data == 'grooming':
        collisions_data = sim_res_folder + 'selfCollisions_data_ball_grooming.pkl'

    if end == 0:
        end = 9.0

    start = int(begin*100)
    stop = int(end*100)
    
    with open(collisions_data, 'rb') as fp:
        data = pickle.load(fp)    

    if sim_data == 'walking':
        title_plot = 'Gait diagram'
        collisions = {'LF':[],'LM':[],'LH':[],'RF':[],'RM':[],'RH':[]}            
        for leg, force in data.items():
            collisions[leg[:2]].append(force.transpose()[0])
            
    elif sim_data == 'grooming':
        title_plot = 'Collisions diagram'        
        collisions = {'LAntenna':[], 'LFTibia':[], 'LFTarsus1':[], 'LFTarsus2':[], 'LFTarsus3':[], 'LFTarsus4':[], 'LFTarsus5':[], 'RFTarsus5':[], 'RFTarsus4':[], 'RFTarsus3':[], 'RFTarsus2':[], 'RFTarsus1':[], 'RFTibia':[], 'RAntenna':[]}#, 'LEye':[], 'REye':[]}        
        for segment, coll in data.items():
            for key, forces in coll.items():
                if segment in collisions.keys():
                    collisions[segment].append([np.linalg.norm(force) for force in forces])        
    
    fig, axs = plt.subplots(len(collisions.keys()), sharex=True,gridspec_kw={'hspace': 0})
    fig.suptitle(title_plot)
    
    for i, (segment, force) in enumerate(collisions.items()):
        sum_force = np.sum(np.array(force), axis = 0)
        segment_force = np.delete(sum_force,0)
        time = np.arange(0,len(segment_force),1)/100
        stance_ind = np.where(segment_force>0)[0]
        if stance_ind.size!=0:
            stance_diff = np.diff(stance_ind)
            stance_lim = np.where(stance_diff>1)[0]
            stance=[stance_ind[0]-1]
            for ind in stance_lim:
                stance.append(stance_ind[ind]+1)
                stance.append(stance_ind[ind+1]-1)
            stance.append(stance_ind[-1])
            start_gait_list = np.where(np.array(stance) >= start)[0]
            if len(start_gait_list)>0:
                start_gait = start_gait_list[0]
            else:
                start_gait = start            
            stop_gait_list = np.where(np.array(stance) <= stop)[0]
            if len(stop_gait_list)>0:
                stop_gait = stop_gait_list[-1]+1
            else:
                stop_gait = start_gait
            stance_plot = stance[start_gait:stop_gait]
            if start_gait%2 != 0:
                stance_plot.insert(0,start)
            if len(stance_plot)%2 != 0:
                stance_plot.append(stop)

            for ind in range(0,len(stance_plot),2):
                axs[i].fill_between(time[stance_plot[ind]:stance_plot[ind+1]], 0, 1, facecolor='black', alpha=1, transform=axs[i].get_xaxis_transform())
        else:
            axs[i].fill_between(time[start:stop], 0, 1, facecolor='white', alpha=1, transform=axs[i].get_xaxis_transform())

        axs[i].set_yticks((0.5,))
        axs[i].set_yticklabels((segment,))

    axs[len(axs)-1].set_xlabel('Time (s)')
    if sim_data == 'walking':
        black_patch = mpatches.Patch(color='black', label='Stance')
    elif sim_data == 'grooming':
        black_patch = mpatches.Patch(color='black', label='Collision')
    axs[0].legend(handles=[black_patch],loc= 'upper right',bbox_to_anchor=(1.1, 1))
    plt.show()
    

def plot_fly_path(begin=0, end=0, sequence=True, save_imgs=False, experiment='', heading=True):
    sim_res_folder = '/home/nely/SimFly/kinematic_matching/results/'
    ball_data = sim_res_folder + 'ballRot_data_ball_walking.pkl'

    with open(ball_data, 'rb') as fp:
        data = pickle.load(fp)

    if end == 0:
        end = len(data)    

    fig = plt.figure()  
    ax = plt.axes()
    m = MarkerStyle(marker=r'$\rightarrow$')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    data_array = np.array(data)
    x_diff = np.diff(data_array.transpose()[0])
    y_diff = np.diff(data_array.transpose()[1])
    #th = -np.diff(data_array.transpose()[2])
    
    x=[0]
    y=[0]
    for i in range(begin,end-1):#, coords in enumerate(x_diff[begin:end]):
        if heading:
            th = -data[i][2]
            x_new = x_diff[i]*np.cos(th) - y_diff[i]*np.sin(th)
            y_new = y_diff[i]*np.cos(th) + x_diff[i]*np.sin(th)
            x.append(x[-1]+x_new)
            y.append(y[-1]+y_new)
        else:
            x.append(data[i][0])
            y.append(data[i][1])

        if sequence:
            print('\rFrame: ' + str(i),end='')
            sc = ax.scatter(x,y,c=np.linspace(begin/100,len(x)/100,len(x)),cmap='winter',vmin=begin/100,vmax=end/100)
        
            m._transform.rotate_deg(th*180/np.pi)        
            ax.scatter(x[-1], y[-1], marker=m, s=200,color='black')
            m._transform.rotate_deg(-th*180/np.pi)
        
            if i == 0:
                sc.set_clim([begin/100,end/100])
                cb = plt.colorbar(sc)
                cb.set_label('Time (s)')

            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            if save_imgs:
                file_name = experiment.split('/')[-1]
                new_folder = 'results/'+file_name.replace('.pkl','/')+'fly_path'
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                name = new_folder + '/img_' + '{:06}'.format(i) + '.jpg'
                fig.set_size_inches(6,4)
                plt.savefig(name, dpi=300)
            else:
                plt.draw()
                plt.pause(0.001)
            ax.clear()

    if not sequence:
        sc = ax.scatter(x,y,c=np.arange(begin/100,end/100,0.01),cmap='winter')
        sc.set_clim([begin/100,end/100])
        cb = plt.colorbar(sc)
        cb.set_label('Time (s)')
        plt.show()

def draw_grf_on_imgs(data_2d, experiment, sim_data='walking', side='L', begin=0, end=0, save_imgs=False,pause=0,grfScalingFactor=10,scale=5):
    data={}
    sim_res_folder = '/home/nely/SimFly/kinematic_matching/results/'
    grf_data = sim_res_folder + 'grf_data_ball_' + sim_data + '.pkl'

    colors = {'LF_leg':(0,0,204),'LM_leg':(51,51,255),'LH_leg':(102,102,255),'RF_leg':(153,76,0),'RM_leg':(255,128,0),'RH_leg':(255,178,102)}

    if end == 0:
        end = 9.0

    start = int(begin*100)
    stop = int(end*100)
    
    with open(grf_data, 'rb') as fp:
        data = pickle.load(fp)
    grf = {'LF':[],'LM':[],'LH':[],'RF':[],'RM':[],'RH':[]}
    angle = {'LF':[],'LM':[],'LH':[],'RF':[],'RM':[],'RH':[]}
    for leg, force in data.items():
        grf[leg[:2]].append(force.transpose()[0])
        angle[leg[:2]].append(force.transpose()[1])

    raw_imgs=[]
    ind_folder = experiment.find('df3d')
    imgs_folder = experiment[:ind_folder-1]
    for frame in range(start,stop):
        if side=='L':
            cam_num = 1
        elif side=='R':
            cam_num = 5
        img_name = imgs_folder + '/camera_' +str(cam_num)+ '_img_' + '{:06}'.format(frame) + '.jpg'
        #print(img_name)
        img = cv.imread(img_name)
        raw_imgs.append(img)

    for i, (leg, force) in enumerate(grf.items()):
        if side in leg:
            sum_force = np.sum(np.array(force), axis = 0)
            leg_force = np.delete(sum_force,0)
            time = np.arange(0,len(leg_force),1)/100
            stance_ind = np.where(leg_force>0)[0]
            if stance_ind.size!=0:
                stance_diff = np.diff(stance_ind)
                stance_lim = np.where(stance_diff>1)[0]
                stance=[stance_ind[0]-1]
                for ind in stance_lim:
                    stance.append(stance_ind[ind]+1)
                    stance.append(stance_ind[ind+1]-1)
                stance.append(stance_ind[-1])
                start_gait_list = np.where(np.array(stance) >= start)[0]
                if len(start_gait_list)>0:
                    start_gait = start_gait_list[0]
                else:
                    start_gait = start            
                stop_gait_list = np.where(np.array(stance) <= stop)[0]
                if len(stop_gait_list)>0:
                    stop_gait = stop_gait_list[-1]+1
                else:
                    stop_gait = start_gait
                stance_plot = stance[start_gait:stop_gait]
                if start_gait%2 != 0:
                    stance_plot.insert(0,start)
                if len(stance_plot)%2 != 0:
                    stance_plot.append(stop)

                for ind in range(0,len(stance_plot),2):
                    for frame in range(stance_plot[ind],stance_plot[ind+1]):
                        x_px = int(data_2d[cam_num][leg+'_leg']['Claw'][frame][0])
                        y_px = int(data_2d[cam_num][leg+'_leg']['Claw'][frame][1])
                        start_pnt = [x_px, y_px]
                        np.array(grf['LF']).transpose()[i+start+1]
                        force_vals = np.array(grf[leg]).transpose()[frame+1]
                        poi = np.where(force_vals>0)[0]
                        if poi.size!=0:
                            mean_angle = np.mean(np.array(angle[leg]).transpose()[frame+1][poi])
                        else:
                            mean_angle = 0
                        #if leg == 'LF':
                        #    print(frame,mean_angle*180/np.pi)
                        
                        force_x = leg_force[frame]*np.cos(mean_angle)*grfScalingFactor
                        force_z = leg_force[frame]*np.sin(mean_angle)*grfScalingFactor
                        h, w, c = raw_imgs[frame-start].shape
                        end_pnt = [x_px+(force_x*h/(2*scale)), y_px-(force_z*h/(2*scale))]
                        color = colors[leg+'_leg']
                        raw_imgs[frame-start] = draw_lines(raw_imgs[frame-start],start_pnt,end_pnt,color=color,thickness=3,arrowHead=True)
                        #coord = (x_px, y_px)   
                        #radius = 8
                        #color = (0, 0, 255) 
                        #thickness = -1
                        #raw_imgs[frame-start] = cv.circle(raw_imgs[frame-start],coord,radius,color,thickness)
    for i, img in enumerate(raw_imgs):
        print('\rFrame: ' + str(i+start),end='')
        if save_imgs:
            file_name = experiment.split('/')[-1]
            new_folder = 'results/'+file_name.replace('.pkl','/')+'grf_on_2d'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            name = new_folder + '/img_' + '{:06}'.format(i) + '.jpg'
            cv.imwrite(name,img)
        else:
            #print(np.array(grf['LF']).transpose()[i+start+1], np.array(angle['LF']).transpose()[i+start+1]*180/np.pi)
            cv.imshow('img',img)
            cv.waitKey(pause)
    cv.destroyAllWindows()


def draw_legs_from_3d(data,exp_dir,data_2d,side='L',dir_name='traking_2d_from_3d_data',begin=0,end=0,saveimgs = False,pixelSize=[5.86e-3,5.86e-3,5.86e-3],pause=0):
    if side == 'L':
        cam_id = 1
    elif side == 'R':
        cam_id = 5
    elif side == 'F':
        cam_id = 3
    
    colors = {'LF_leg':(0,0,204),'LM_leg':(51,51,255),'LH_leg':(102,102,255),'RF_leg':(153,76,0),'RM_leg':(255,128,0),'RH_leg':(255,178,102)}

    if end == 0:
        try:
            end = len(data['LF_leg']['Coxa']['raw_pos_aligned'])
        except:
            end = len(data['LF_leg']['Coxa']['raw_pos'])
    for frame in range(begin,end):
        print('\rFrame: ' + str(frame),end='')
        df3d_dir = exp_dir.find('df3d')
        folder = exp_dir[:df3d_dir]
        img_name = folder + 'camera_' + str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
        img = cv.imread(img_name)
        if side != 'F':
            offset = np.mean(data_2d[cam_id][side+'M_leg']['Coxa'],axis=0)
        else:
            offset_L = np.mean(data_2d[1]['LM_leg']['Coxa'],axis=0)[1]
            offset_R = np.mean(data_2d[5]['RM_leg']['Coxa'],axis=0)[1]
        for leg, body_parts in data.items():
            if side in leg[:2] or side =='F':#'L' or leg[0]=='R':
                color = colors[leg]
                for segment, metrics in body_parts.items():
                    for metric, points in metrics.items():
                        if 'raw_pos' in metric and segment != 'Coxa':
                            if 'Femur' in segment:
                                start = data[leg]['Coxa']['fixed_pos_aligned']/np.array(pixelSize)
                                end = points[frame]/np.array(pixelSize)
                            if 'Tibia' in segment:
                                start = data[leg]['Femur'][metric][frame]/np.array(pixelSize)
                                end = points[frame]/np.array(pixelSize)
                            if 'Tarsus' in segment:
                                start = data[leg]['Tibia'][metric][frame]/np.array(pixelSize)
                                end = points[frame]/np.array(pixelSize)
                            if 'Claw' in segment:
                                start = data[leg]['Tarsus'][metric][frame]/np.array(pixelSize)
                                end = points[frame]/np.array(pixelSize)
                            if side == 'L':
                                start_point = [int(start[0]),int(-start[2])]+offset
                                end_point = [int(end[0]),int(-end[2])]+offset
                            elif side == 'R':
                                start_point = [int(-start[0]),int(-start[2])]+offset
                                end_point = [int(-end[0]),int(-end[2])]+offset
                            elif side == 'F':
                                h, w, c = img.shape
                                if 'L' in leg:
                                    start_point = [int(start[1]+w/2),int(offset_L-start[2])]
                                    end_point = [int(end[1]+w/2),int(offset_L-end[2])]
                                if 'R' in leg:
                                    start_point = [int(start[1]+w/2),int(offset_R-start[2])]
                                    end_point = [int(end[1]+w/2),int(offset_R-end[2])]
                            img = draw_lines(img,start_point,end_point,color=color)
        if saveimgs:
            file_name = exp_dir.split('/')[-1]
            new_folder = 'results/'+file_name.replace('.pkl','/')+dir_name+'_'+side+'/'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            name = new_folder + 'camera_' + str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
            cv.imwrite(name,img)
        cv.imshow('img',img)
        cv.waitKey(pause)
    cv.destroyAllWindows()

def draw_lines(img, start, end, color = (255, 0, 0), thickness=5, arrowHead=False):
    coords_prev = np.array(start).astype(int)
    coords_next = np.array(end).astype(int)

    start_point = (coords_prev[0],coords_prev[1])
    end_point = (coords_next[0],coords_next[1]) 

    if arrowHead:
        if np.linalg.norm(coords_prev-coords_next)>100:
            tL = 0.1
        else:
            tL = 0.5
        cv.arrowedLine(img, start_point, end_point, color, thickness, tipLength = tL)
    else:
        cv.line(img, start_point, end_point, color, thickness) 

    return img

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
        #ax_3d.set_xlim(-2.1, 3.5)
        #ax_3d.set_ylim(-2.1, 3)
        #ax_3d.set_zlim(0.1, 2.3)
        ax_3d.grid(True)

        if savePlot:
            folder = 'results/tracking_3d_leftside/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            figName = folder + 'pos_3d_frame_'+str(frame)+'.png'
            fig_3d.savefig(figName)


        ax_2d.legend()
        ax_2d.set_xlabel(plane[0]+' (mm)')
        ax_2d.set_ylabel(plane[1]+' (mm)')
        ax_2d.grid(True)

        plt.show()

def plot_fixed_coxa(aligned_dict,plane='xy'):
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
        if plane == 'xy':
            ax_2d.plot(x, y, '--x', label=leg)
        if plane == 'xz':
            ax_2d.plot(x, z, '--x', label=leg)
        if plane == 'yz':
            ax_2d.plot(y, z, '--x', label=leg)

    ax_3d.legend()
    ax_3d.set_xlabel('X Label')
    ax_3d.set_ylabel('Y Label')
    ax_3d.set_zlabel('Z Label')
    ax_3d.grid(True)

    ax_2d.legend()
    ax_2d.set_xlabel(plane[0]+' (mm)')
    ax_2d.set_ylabel(plane[1]+' (mm)')
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


def plot_legs_from_angles(angles,data_dict,exp_dir,begin=0,end=0,plane='xz',saveImgs = False, dir_name='km', extraDOF = {}, ik_angles = False, pause = False,lim_axes=True,offset=0):

    #colors_real= {'LF_leg':(1,0,0),'LM_leg':(0,1,0),'LH_leg':(0,0,1),'RF_leg':(1,1,0),'RM_leg':(1,0,1),'RH_leg':(0,1,1)}
    #colors = {'LF_leg':(1,0.5,0.5),'LM_leg':(0.5,1,0.5),'LH_leg':(0.5,0.5,1),'RF_leg':(1,1,0.5),'RM_leg':(1,0.5,1),'RH_leg':(0.5,1,1)}

    colors = {'LF_leg':(204/255,0,0),'LM_leg':(1,51/255,51/255),'LH_leg':(1,102/255,102/255),'RF_leg':(0,76/255,153/255),'RM_leg':(0,0.5,1),'RH_leg':(102/255,178/255,1)}

    
    legend_3D = [(Line2D([0], [0], marker='o', ms=6, ls='-', lw=2, color=colors['LF_leg']),
                  Line2D([0], [0], marker='o', ms=6, ls='-', lw=2, color=colors['RF_leg'])),
                 (Line2D([0], [0], marker='o', ms=6, ls='-', lw=2, color=colors['LM_leg']),
                  Line2D([0], [0], marker='o', ms=6, ls='-', lw=2, color=colors['RM_leg'])),
                 (Line2D([0], [0], marker='o', ms=6, ls='-', lw=2, color=colors['LH_leg']),
                  Line2D([0], [0], marker='o', ms=6, ls='-', lw=2, color=colors['RH_leg']))]

    legend_FK = [(Line2D([0], [0], marker='x', ms=6, ls='--', lw=2, color=colors['LF_leg']),
                  Line2D([0], [0], marker='x', ms=6, ls='--', lw=2, color=colors['RF_leg'])),
                 (Line2D([0], [0], marker='x', ms=6, ls='--', lw=2, color=colors['LM_leg']),
                  Line2D([0], [0], marker='x', ms=6, ls='--', lw=2, color=colors['RM_leg'])),
                 (Line2D([0], [0], marker='x', ms=6, ls='--', lw=2, color=colors['LH_leg']),
                  Line2D([0], [0], marker='x', ms=6, ls='--', lw=2, color=colors['RH_leg']))]

    fig_3d = plt.figure()
    if plane is not '':
        view_2d = plt.figure()
    
    if end == 0:
        end = len(angles['LF_leg']['yaw'])

    order = ['RH_leg','RM_leg','RF_leg','LH_leg','LM_leg','LF_leg']
    angles_rev = {leg:angles[leg] for leg in order}

    lim_x = (-2,3)
    lim_y = (-2,2)
    lim_z = (-1.5,0.5)
    
    for frame in range(begin, end):
        print('\rFrame: '+str(frame),end='')
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_xlabel('mm')
        ax_3d.set_ylabel('mm')
        ax_3d.set_zlabel('mm')
        ax_3d.grid(True)
        ax_3d.view_init(40, -45)

        if lim_axes:
            ax_3d.set_xlim(lim_x)
            ax_3d.set_ylim(lim_y)
            ax_3d.set_zlim(lim_z)

        if plane is not '':
            ax_2d = plt.axes()
            ax_2d.set_xlabel(plane[0])
            ax_2d.set_ylabel(plane[1])
            ax_2d.grid(True)

        for name, leg in angles_rev.items():
            coxa_pos = data_dict[name]['Coxa']['fixed_pos_aligned']
            real_pos_femur = data_dict[name]['Femur']['raw_pos_aligned'][frame]
            real_pos_tibia = data_dict[name]['Tibia']['raw_pos_aligned'][frame]
            real_pos_tarsus = data_dict[name]['Tarsus']['raw_pos_aligned'][frame]
            real_pos_claw = data_dict[name]['Claw']['raw_pos_aligned'][frame]

            extraDOF_vals = {}
            for name_DOF, dict_DOF in extraDOF.items():
                try:
                    extraDOF_vals[name_DOF] = dict_DOF[name][name_DOF][frame]
                except:
                    extraDOF_vals[name_DOF] = dict_DOF[name][name_DOF]['best_angle'][frame]
            if ik_angles:
                pos_3d = fk_from_ik(leg, name, data_dict, frame).transpose()
            else:
                pos_3d = utils_angles.calculate_forward_kinematics(name, frame, leg, data_dict, extraDOF=extraDOF_vals,ik_angles=ik_angles,offset=offset).transpose()

                      
            x = pos_3d[0]
            y = pos_3d[1]
            z = pos_3d[2]
            ax_3d.plot(x, y, z, '--x', label=name, color = colors[name])            

            real_pos_3d = np.array([coxa_pos,real_pos_femur,real_pos_tibia,real_pos_tarsus,real_pos_claw]).transpose()
            real_x = real_pos_3d[0]
            real_y = real_pos_3d[1]
            real_z = real_pos_3d[2]
            ax_3d.plot(real_x, real_y, real_z, '-o', label=name+'_real', color = colors[name])

            if plane == 'xy':
                if lim_axes:
                    ax_2d.set_xlim(lim_x)
                    ax_2d.set_ylim(lim_y)
                ax_2d.plot(real_x, real_y, '-o', label=name+'_real', color = colors[name])
                ax_2d.plot(x, y, '--x', label=name, color = colors[name])
            if plane == 'xz':
                if lim_axes:
                    ax_2d.set_xlim(lim_x)
                    ax_2d.set_ylim(lim_z)
                ax_2d.plot(real_x, real_z, '-o', label=name+'_real', color = colors[name])
                ax_2d.plot(x, z, '--x', label=name, color = colors[name])
            if plane == 'yz':
                if lim_axes:
                    ax_2d.set_xlim(lim_y)
                    ax_2d.set_ylim(lim_z)
                ax_2d.plot(real_y, real_z, '-o', label=name+'_real', color = colors[name])
                ax_2d.plot(y, z, '--x', label=name, color = colors[name])
            
        if saveImgs:
            file_name = exp_dir.split('/')[-1]
            new_folder_3d = 'results/'+file_name.replace('.pkl','/')+dir_name+'_3d/'
            new_folder_2d = 'results/'+file_name.replace('.pkl','/')+dir_name+'_'+plane+'/'
            if not os.path.exists(new_folder_3d):
                os.makedirs(new_folder_3d)
            if plane is not '':
                if not os.path.exists(new_folder_2d):
                    os.makedirs(new_folder_2d)
            name_3d = new_folder_3d + 'km_3d' + '_img_' + '{:06}'.format(frame) + '.jpg'
            name_2d = new_folder_2d + 'km_' + plane + '_img_' + '{:06}'.format(frame) + '.jpg'
            
            w_img = int(12)
            h_img = int(w_img*3/4)
            fig_3d.set_size_inches(w_img, h_img)

            first_legend = ax_3d.legend(legend_3D,['Prothoracic legs','Mesothoracic legs','Metathoracic legs'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},handlelength=6,loc='lower left',title='Pose from 3D tracking',bbox_to_anchor=(0, -0.03))

            ax_legend = plt.gca().add_artist(first_legend)

            ax_3d.legend(legend_FK,['Prothoracic legs','Mesothoracic legs','Metathoracic legs'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},handlelength=6,loc='lower right',title='Pose from forward kinematics',bbox_to_anchor=(1.03, -0.03))

            #ax_3d.legend(handles=legend_elements, loc='upper right',ncol=2,handletextpad=0.1)
            
            fig_3d.savefig(name_3d,dpi=300,bbox_inches='tight')
            if plane is not '':
                view_2d.savefig(name_2d)

        if not saveImgs:
            ax_3d.legend()
            if plane is not '':
                view_2d.legend()
            if pause:
                plt.show()
            else:
                #plt.show(block=False)
                plt.draw()
                plt.pause(0.001)

        fig_3d.clf()
        if plane is not '':
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

def plot_error(errors_dict,begin=0,end=0,name='filename.png',dpi=300,save=False,BL=2.88):
    legs = list(errors_dict.keys())
    angles = list(errors_dict['LF_leg'].keys())
    df_errors = pd.DataFrame()

    colors = [(204/255,0,0),(1,51/255,51/255),(1,102/255,102/255),(0,76/255,153/255),(0,0.5,1),(102/255,178/255,1)]

    if end == 0:
        end = len(errors_dict[legs[0]][angles[0]]['min_error'])
        
    for leg in legs:
        for angle in angles:
            vals = []
            for err in errors_dict[leg][angle]['min_error'][begin:end]:
                mae = err[0]/(len(err)-1)
                norm_error = mae/BL
                vals.append(norm_error)

            df_vals = pd.DataFrame(vals,columns=['norm_error'])
            df_vals['leg'] = leg
            df_vals['angle'] = angle

            df_errors = df_errors.append(df_vals, ignore_index = True)
    
    for angle1 in angles:
        x1 = df_errors['norm_error'].loc[df_errors['angle']==angle1]
        print(angle1 + ' mean/std = ' + str(np.mean(x1)) + ' /+- ' + str(np.std(x1)))
        for angle2 in angles[angles.index(angle1)+1:]:
        #if angle != 'base':            
            x2 = df_errors['norm_error'].loc[df_errors['angle']==angle2]
            ztest , pval = stests.ztest(x1, x2=x2, value=0, alternative='two-sided')

            print(angle1 + ' vs ' + angle2 + ': ', ztest, pval,)
            if pval > 0.001:
                print(angle1 + " is not statistically different from " + angle2)
        print()
    
    ax = sns.violinplot(x='angle', y='norm_error', data=df_errors, color="0.8")
    for violin, alpha in zip(ax.collections[::2], [0.7]*len(angles)):
        violin.set_alpha(alpha)
    ax = sns.stripplot(x='angle', y='norm_error', hue='leg', data=df_errors, jitter=True, zorder=0, size=3)
    #ax = sns.swarmplot(x='angle', y='norm_error', hue='leg', data=df_errors, zorder=0, size=3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
    plt.title('Comparison adding an extra DOF')
    
    figure = plt.gcf()  # get current figure
    w_img = int(len(angles)*1.5)
    h_img = int(w_img*3/4)
    figure.set_size_inches(w_img, h_img) # set figure's size manually to your full screen (32x18)
    if save:
        plt.savefig(name, dpi=dpi,bbox_inches='tight')
    plt.show()
    
    return df_errors

def calculate_inverse_kinematics(data_dict, init_angles={},roll_tr=False):
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
