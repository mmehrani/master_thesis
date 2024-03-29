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
total_time = 200
start_time_to_sample = 100
g = 10
delay_time = 1.5
# g = 0

sample_network = Animated_network_of_neurons(neuron_model,
                                              num_neurons, g = g,
                                              random_input_span = (9.5,13.5), alpha = 20)
# sample_network = Animated_network_of_neurons(neuron_model,
#                                               num_neurons, g = g,
#                                               random_input_span = (1.2,2.8), alpha = 20)
# sample_network = Animated_network_of_neurons(neuron_model,
#                                               num_neurons, g = g, alpha = 20)


sample_network.brace_for_lunch(total_time, time_step = 0.01, delay_time = delay_time)

for i in tqdm(range( int( start_time_to_sample / sample_network.time_step ) ) ):
    sample_network._march_on(i)

epsilon = 0
for i in tqdm(range( int( start_time_to_sample / sample_network.time_step ) ) ):
    sample_network.e_arr[i] = sample_network.e_arr[i] + epsilon
    # sample_network.spiking_records[i] = sample_network.spiking_records[i] + 10

sample_network.render_animation(start_time_to_sample, show_space=True, show_pop = False,
                                show_field= True, show_velocity=True)

# version_name = 'black_white'
# path = os.path.join('animations','sea_shore',version_name,
#                     "N{}_g{}_d{}_Imin{}_Imax{}_{}.gif".format(
#                         num_neurons, g, delay_time,
#                         sample_network.random_input_span[0],sample_network.random_input_span[1],
#                         neuron_model))

# sample_network.render_animation(start_time_to_sample, show_space=True, show_pop = False,
#                                 show_field= True, show_velocity=True, path = path)


