# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:14:03 2021

@author: mohsen
"""


import os
from network_reference import Animated_network_of_neurons
from tqdm import tqdm
import numpy as np

current_models = ['IF','Rotational','Non_repulsive_rotational']
neuron_model = current_models[2]

with open("network_reference.py") as net_ref: 
    lines = net_ref.readlines() #read 
 
    #modify 
    lines[0] = "neuron_engine = '{}'\n".format(neuron_model) #you can replace zero with any line number. 
 
with open("network_reference.py", "w") as net_ref: 
    net_ref.writelines(lines) #write back 


    
num_neurons = 10000
total_time = 120
start_time_to_sample = 100
g = 0
# g = 0

sample_network = Animated_network_of_neurons('Non_repulsive_rotational',
                                              num_neurons, g = g,random_input_span = (9.5,9.5))
# sample_network = Animated_network_of_neurons('Non_repulsive_rotational',
#                                              num_neurons, g = g,random_input_span = (9.5,13.5),
#                                              xlim = [0,100],ylim = [np.pi-1,np.pi])
sample_network.brace_for_lunch(total_time, time_step = 0.01, delay_time = 0.1)


for i in tqdm(range( int( start_time_to_sample / sample_network.time_step ) ) ):
    sample_network._march_on(i)


sample_network.render_animation()
# version_name = 'well_in_negatives'
# path = os.path.join('animations','sea_shore',version_name,"N{}_g{}_Imin{}_Imax{}_neurons_rotational.html".format(
#     num_neurons,g,sample_network.random_input_span[0],sample_network.random_input_span[1]))
# ani.save(path, writer='imagemagick')

# with open(path, "w") as f:
#     print(ani.to_html5_video(), file=f)

