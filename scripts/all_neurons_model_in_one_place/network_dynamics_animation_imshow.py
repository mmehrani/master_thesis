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


class Animated_network_of_neurons(Network_of_neurons):
    def brace_for_lunch(self,random_input_span,total_time,time_step = 0.001,delay_time = 0.1):
        
        # random_input_span = (3.5,13.5)
        
        total_steps = int(total_time/time_step)
        time_span = np.arange(0,total_time,time_step)
        
        self.total_time = total_time
        self.total_steps = total_steps
        self.time_step = time_step
        self.delay_step = int(delay_time/time_step)
        
        
        self.m_arr = np.zeros(total_steps)
        self.e_arr = np.zeros(total_steps)
        self.random_input = np.random.uniform(*random_input_span,size = self.num_neurons)
        
        self.amin_saman_param = np.zeros( total_steps )
        self.spiking_records = np.zeros(total_steps)
        
        self._set_cornometers()
    
        return

    pass



num_neurons = 10000
total_time = 110
start_time_to_sample = 100
g = 12.7
# g = 0

sample_network = Animated_network_of_neurons(num_neurons, g = g)
# random_input_span = (3.5,13.5)
random_input_span = (9.5,13.5)
# random_input_span = (2.7,2.8)
sample_network.brace_for_lunch(random_input_span, total_time, time_step = 0.01, delay_time = 0.1)


for i in tqdm(range( int( start_time_to_sample / sample_network.time_step ) ) ):
    sample_network._march_on(i)


current_sort_args = np.argsort(sample_network.random_input)


max_degree = np.pi
min_degree = -5*np.pi/2

# max_degree = -0.5
# min_degree = 1

extent = [1 , num_neurons, max_degree , min_degree] #imshow axises are updside down

grating_num = 100
grating_blocks_length = ( max_degree - min_degree )/grating_num

plateau = np.zeros((grating_num, num_neurons))

color_num = 5

def init():
    phase_marks = np.floor( (sample_network.potentail_arr[current_sort_args] - min_degree) / grating_blocks_length ) #sorted neurons
    for neuron_index in range(num_neurons):
        plateau[int(phase_marks[neuron_index]),neuron_index] = color_num
    colored_plateau.set_data(plateau)
    
    e_pulse.set_ydata(time_series*0)
    return


def update(frame):
    
    was_network_active = np.sum(sample_network.spike_mask) > 0
    
    sample_network._march_on(frame-1)
    phase_marks = np.floor( (sample_network.potentail_arr[current_sort_args] - min_degree) / grating_blocks_length ) #sorted neurons
    
    # Change the spiking group color if they stopped spiking
    if np.sum(sample_network.spike_mask) == 0 and was_network_active:
        global color_num
        color_num += 2
        color_num = (color_num %10)

    # Update neuron phase marks
    for neuron_index in range(num_neurons):
        current_color = np.sum( plateau[:,neuron_index] )
        plateau[:,neuron_index] = plateau[:,neuron_index]*0
        plateau[int(phase_marks[neuron_index]),neuron_index] = current_color
        
        #Spiking ones
        if sample_network.spike_mask[current_sort_args[neuron_index]] == True:
            plateau[int(phase_marks[neuron_index]),neuron_index] = color_num
        
    
    colored_plateau.set_data(plateau)
    # colored_pop_dist.set_data( np.log10( np.atleast_2d(np.sum(plateau>0,axis = 1)) ).T )
    pop_dist.set_xdata( np.sum(plateau>0,axis = 1) )
    e_pulse.set_ydata(sample_network.e_arr[frame-80:frame])
    wind_direction.set_ydata(sample_network.driving_wind[current_sort_args])
    return plateau


gs = gridspec.GridSpec(3, 2, width_ratios = (10,4), height_ratios = (10,5,2), wspace = 0.2)

# fig = plt.figure(figsize = (13.8,7.2),dpi = 100)
fig = plt.figure()

ax = fig.add_subplot(gs[0,0])
ax_stat = fig.add_subplot(gs[0,1], sharey = ax)
ax_e = fig.add_subplot(gs[2, 0])
ax_theta_dot = fig.add_subplot(gs[1, 0], sharex = ax)

y_label_list = [r'$-5\frac{\pi}{2}$', '$-\pi$', '0', '$\pi$']
ax.set_yticks([min_degree, - np.pi, 0, max_degree])
ax.set_yticklabels(y_label_list)
ax.set_title('Network dynamic N={} g={}'.format(num_neurons,g))
colored_plateau = ax.imshow( plateau, aspect= 'auto', extent = extent , vmin = 0, vmax = 10, cmap = 'tab20b')

ax_e.set_ylabel('E')
ax_e.set_xlabel('t')

ax_theta_dot.set_ylabel(r'$\dot\theta$')
ax_theta_dot.set_xlabel('neuron number')
wind_direction, = ax_theta_dot.plot(range(1,num_neurons+1), sample_network.driving_wind[current_sort_args])
ax_theta_dot.set_ylim([-13,13])

ax_e.set_xlim([0,1])
ax_e.set_ylim([0,1.5])
time_series = np.arange(0,0.8,sample_network.time_step)
e_pulse, = ax_e.plot(time_series,time_series*0)

ax_stat.set_xlabel('population')
pop_dist, = ax_stat.plot( np.sum(plateau>0,axis = 1), np.linspace(min_degree,max_degree,num = grating_num) )
ax_stat.set_xlim([0,3*num_neurons/grating_num])

fig.tight_layout()

frames_range = range( int(start_time_to_sample/sample_network.time_step), sample_network.total_steps)
ani = FuncAnimation(fig, update,init_func = init, frames= frames_range, interval = 50)


version_name = 'well_in_negatives'
path = os.path.join('animations','sea_shore',version_name,"N{}_g{}_Imin{}_Imax{}_neurons_rotational.html".format(num_neurons,g,random_input_span[0],random_input_span[1]))
# ani.save(path, writer='imagemagick')

with open(path, "w") as f:
    print(ani.to_html5_video(), file=f)

