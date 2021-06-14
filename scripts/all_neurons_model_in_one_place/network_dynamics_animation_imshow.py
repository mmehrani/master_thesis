# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:14:03 2021

@author: mohsen
"""


import os
from network_reference import Network_of_neurons
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

from matplotlib import animation, rc
from IPython.display import HTML
from tqdm import tqdm
import progressbar


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



num_neurons = 1000
total_time = 100
start_time_to_sample = 90
g = 60


sample_network = Animated_network_of_neurons(num_neurons, g = g)
random_input_span = (3.5,13.5)
sample_network.brace_for_lunch(random_input_span, total_time, time_step = 0.01)


for i in tqdm(range( int( start_time_to_sample / sample_network.time_step ) ) ):
    sample_network._march_on(i)


current_sort_args = np.argsort(sample_network.random_input)


max_degree = np.pi
min_degree = -5*np.pi/2

extent = [1 , num_neurons, max_degree , min_degree] #imshow axises are updside down

grating_num = 100
grating_blocks_length = ( max_degree - min_degree )/grating_num

plateau = np.zeros((grating_num, num_neurons))


def update(frame):
    fig.suptitle('Network dynamic N={} g={}'.format(num_neurons,g))
    sample_network._march_on(frame-1)
    phase_marks = np.floor( (sample_network.potentail_arr - min_degree) / grating_blocks_length )
    
    for neuron_index in range(num_neurons):
        sample_zero = np.zeros(grating_num)
        sample_zero[int(phase_marks[neuron_index])] = 2
        plateau[:,neuron_index] = sample_zero
    
    colored_plateau.set_data(plateau)

    return plateau

fig, ax = plt.subplots()
colored_plateau = ax.imshow( plateau, aspect= 'auto', extent = extent, vmin = 0, vmax = 1 )


y_label_list = [r'$-5\frac{\pi}{2}$', '$-\pi$', '0', '$\pi$']
ax.set_yticks([min_degree, - np.pi, 0, max_degree])
ax.set_yticklabels(y_label_list)

fig.colorbar(colored_plateau)

frames_range = range( int(start_time_to_sample/sample_network.time_step), sample_network.total_steps)
ani = FuncAnimation(fig, update, frames= frames_range, interval = 50)


version_name = 'well_in_negatives'
path = os.path.join('animations','sea_shore',version_name,"{}_{}_neurons_rotational.gif".format(num_neurons,g))
# ani.save(path, writer='imagemagick')

