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


class Two_connected_neural_network():
    def __init__(self, first_Animated_network_of_neurons, second_Animated_network_of_neurons, inter_connectivity):
        self.first_network = first_Animated_network_of_neurons
        self.second_network = second_Animated_network_of_neurons
        self.connectivity = inter_connectivity
        
        self.first_network.external_input_strength = inter_connectivity
        self.second_network.external_input_strength = inter_connectivity
        
        self.first_network.external_input = self.second_network.e_arr
        self.second_network.external_input = self.first_network.e_arr
        return
    
    # def ignite(self, checkpoint_step):
        # for i in tqdm(range( int( 10 / self.first_network.time_step ) ) ):
        #     self.first_network._march_on(i)
        #     # self.second_network._march_on(i)
            
        # for i in tqdm(range( int( 10 / self.first_network.time_step ), int( checkpoint_step / self.first_network.time_step ) ) ):
        #     self.first_network._march_on(i)
        #     self.second_network._march_on(i - int( 10 / self.first_network.time_step ))
        # return

    def ignite(self, checkpoint_step):
        for i in tqdm(range( int( checkpoint_step / self.first_network.time_step ) ) ):
            self.first_network._march_on(i)
            self.second_network._march_on(i)
        return
    
    def get_networks_animation(self):
        self.first_network.render_animation(start_time_to_sample, show_space=True, show_pop = False,
                                            show_field= True, show_velocity=False)
        self.second_network.render_animation(start_time_to_sample, show_space=True, show_pop = False,
                                            show_field= True, show_velocity=False)
        return
    pass

num_neurons = 10000
total_time = 200
start_time_to_sample = 100
g = 10
between_g =  - 1
delay_time = 0.5
# g = 0

sample_network_one = Animated_network_of_neurons(neuron_model,
                                                 num_neurons, g = g,
                                                 random_input_span = (9.5,13.5), alpha = 20)
sample_network_one.potentail_arr -= 2

sample_network_two = Animated_network_of_neurons(neuron_model,
                                                 num_neurons, g = g,
                                                 random_input_span = (9.5,13.5), alpha = 20)



sample_network_one.brace_for_lunch(total_time, time_step = 0.01, delay_time = delay_time)
sample_network_two.brace_for_lunch(total_time, time_step = 0.01, delay_time = delay_time)

total_network = Two_connected_neural_network(sample_network_one, sample_network_two,  between_g)
total_network.ignite(start_time_to_sample)
total_network.get_networks_animation()
