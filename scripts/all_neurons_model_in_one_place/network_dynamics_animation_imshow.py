# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:14:03 2021

@author: mohsen
"""


import os
from network_reference import Network_of_neurons
import numpy as np
import networkx as nx

from matplotlib import animation, rc
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import HTML
from tqdm import tqdm

from scipy.stats import rankdata

current_models = ['IF','Rotational','Non_repulsive_rotational']
neuron_model = current_models[2]

with open("network_reference.py") as net_ref: 
    lines = net_ref.readlines() #read 
 
    #modify 
    lines[0] = "neuron_engine = '{}'\n".format(neuron_model) #you can replace zero with any line number. 
 
with open("network_reference.py", "w") as net_ref: 
    net_ref.writelines(lines) #write back 

class Animated_network_of_neurons(Network_of_neurons):
    def __init__(self,num_neurons,g,alpha = 20):
        super().__init__(num_neurons,g,alpha = 20)
        # self.potentail_arr = np.random.uniform(-np.pi,np.pi, size = num_neurons)
        self.potentail_arr = np.tile( np.linspace(-np.pi,np.pi, num = 100), reps = 100 )
        if neuron_model == current_models[0]:
            self.random_input_span = (1.2,2.8)
            
            self.ceiling_state = 1
            self.floor_state =  - 0.5 #used for animations frame
            
            self.important_states_namestrings = [r'-0.5', '0', '0', str(self.ceiling_state)]
            
            self.wind_name = r'$\dot v$'
            self.wind_amplitude = [-1,1]
        
        elif neuron_model in current_models[1:3]:
            self.random_input_span = (9.5,13.5)
            self.ceiling_state = np.pi
            self.floor_state = - 5*np.pi/2 #used for animations frame
            
            self.important_states_namestrings = [r'$-5\frac{\pi}{2}$', '$-\pi$', '0', '$\pi$']            
            
            self.wind_name = r'$\dot\theta$'
            self.wind_amplitude = [-13,13]
        
        
            
    def brace_for_lunch(self,total_time,time_step = 0.01,delay_time = 0.1):
        
        # random_input_span = (3.5,13.5)
        
        total_steps = int(total_time/time_step)
        time_span = np.arange(0,total_time,time_step)
        
        self.total_time = total_time
        self.total_steps = total_steps
        self.time_step = time_step
        self.delay_step = int(delay_time/time_step)
        
        
        self.m_arr = np.zeros(total_steps)
        self.e_arr = np.zeros(total_steps)
        # self.random_input = np.random.uniform(*self.random_input_span,size = self.num_neurons)
        
        self.random_input = np.linspace(self.random_input_span[0], self.random_input_span[1], num=100)
        self.random_input = np.repeat(self.random_input, repeats = int( self.num_neurons / len(self.random_input)) )
        
        self.amin_saman_param = np.zeros( total_steps )
        self.spiking_records = np.zeros(total_steps)
        
        self._set_cornometers()
    
        return

    pass



    
num_neurons = 10000
total_time = 60
start_time_to_sample = 0
g = 10
# g = 0

sample_network = Animated_network_of_neurons(num_neurons, g = g)

sample_network.brace_for_lunch(total_time, time_step = 0.01, delay_time = 0.1)


# for i in tqdm(range( int( start_time_to_sample / sample_network.time_step ) ) ):
#     sample_network._march_on(i)

extent = [1 , num_neurons, sample_network.ceiling_state , sample_network.floor_state] #imshow axises are updside down

warp_num = 100 #vertical axis
weft_num = 100 #horizental axix


column_indices = np.floor( (sample_network.random_input - sample_network.random_input_span[0])/(sample_network.random_input_span[1] - sample_network.random_input_span[0]) * (weft_num-1) ).astype('int')
argsort_inputs = np.argsort(column_indices)

grating_blocks_length = ( sample_network.ceiling_state - sample_network.floor_state )/warp_num
plateau = np.zeros((warp_num, weft_num))

color_num = 5
global color_marks
color_marks = np.ones(num_neurons) * color_num

def init():
    
    global colored_plateau, pop_dist, e_pulse, wind_direction
    
    ax = fig.add_subplot(gs[0,0])
    ax_stat = fig.add_subplot(gs[0,1], sharey = ax)
    ax_e = fig.add_subplot(gs[2, 0])
    ax_theta_dot = fig.add_subplot(gs[1, 0], sharex = ax)
    
    
    ax.set_yticks([sample_network.floor_state, - np.pi, 0, sample_network.ceiling_state])
    ax.set_yticklabels(sample_network.important_states_namestrings)
    ax.set_title('Network dynamic N={} g={}'.format(num_neurons,g))
    # colored_plateau = ax.imshow( plateau, aspect= 'auto', extent = extent , vmin = 0, vmax = 10, cmap = 'tab20b')
    colored_plateau = ax.imshow( plateau, aspect= 'auto', extent = extent , vmin = 0, vmax = 10, cmap = 'hot')
    ax.invert_yaxis()
    
    ax_e.set_ylabel('E')
    ax_e.set_xlabel('t')
    
    ax_theta_dot.set_ylabel(sample_network.wind_name)
    ax_theta_dot.set_xlabel('neuron number')
    wind_direction, = ax_theta_dot.plot(range(1,num_neurons+1), sample_network.driving_wind[argsort_inputs])
    # wind_direction, = ax_theta_dot.plot(range(1,num_neurons+1), np.zeros(num_neurons))
    ax_theta_dot.set_ylim(sample_network.wind_amplitude)
    
    ax_e.set_xlim([0,1])
    ax_e.set_ylim([-0.5,2.5])
    time_series = np.arange(0,0.8,sample_network.time_step)
    e_pulse, = ax_e.plot(time_series,time_series*0)
    
    ax_stat.set_xlabel('population')
    pop_dist, = ax_stat.plot( np.sum(plateau>0,axis = 1), np.linspace(sample_network.floor_state,sample_network.ceiling_state,num = warp_num) )
    ax_stat.set_xlim([0,num_neurons/warp_num])

    return


def update(frame):
    
    was_network_active = np.sum(sample_network.spike_mask) > 0
    
    plateau[:]=plateau[:]*0
    sample_network._march_on(frame)
    phase_marks = np.floor( (sample_network.potentail_arr - sample_network.floor_state) / grating_blocks_length ).astype('int') #sorted neurons
    
    # Change the spiking group color if they stopped spiking
    if np.sum(sample_network.spike_mask) == 0 and was_network_active:
        global color_num
        color_num += 2
        color_num = (color_num %10)

    # Update neuron phase marks and color
    for neuron_index in range(num_neurons):
        neuron_column = column_indices[neuron_index]
        neuron_mark = phase_marks[neuron_index]
        # plateau[:,neuron_column] = plateau[:,neuron_column]*0
        
        #Spiking ones
        # if sample_network.spike_mask[neuron_index] == True:
        #     color_marks[neuron_index] = color_num
            
        #coloring
        if neuron_mark >= 0 and neuron_mark <= warp_num:
            plateau[neuron_mark,neuron_column] = color_marks[neuron_index]
        
    
    colored_plateau.set_data(plateau)
    # colored_pop_dist.set_data( np.log10( np.atleast_2d(np.sum(plateau>0,axis = 1)) ).T )
    pop_dist.set_xdata( np.sum(plateau>0,axis = 1) )
    e_pulse.set_ydata( np.roll( sample_network.e_arr, shift= 80 - frame )[0:80] ) #takes care of initial steps
    wind_direction.set_ydata(sample_network.driving_wind[argsort_inputs])
    return plateau


gs = gridspec.GridSpec(3, 2, width_ratios = (10,4), height_ratios = (10,5,2), wspace = 0.2)

# fig = plt.figure(figsize = (13.8,7.2),dpi = 100)
fig = plt.figure()
plt.rc('font', family='serif')
plt.style.use('dark_background')


# fig.tight_layout()

frames_range = range( int(start_time_to_sample/sample_network.time_step) + 1, sample_network.total_steps)
ani = FuncAnimation(fig, update,init_func = init, frames= frames_range, interval = 50)

version_name = 'well_in_negatives'
path = os.path.join('animations','sea_shore',version_name,"N{}_g{}_Imin{}_Imax{}_neurons_rotational.html".format(
    num_neurons,g,sample_network.random_input_span[0],sample_network.random_input_span[1]))
# ani.save(path, writer='imagemagick')

# with open(path, "w") as f:
#     print(ani.to_html5_video(), file=f)

