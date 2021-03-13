# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Rotational_neural_network:
    
    def __init__(self,num_neurons,g,alpha = 20):
        self.num_neurons = num_neurons
        self.g = g
        self.alpha = alpha
    
        return
    
    def _retarded_spikes_record(self,step):
        
        if step > self.delay_step: 
            return self.spiking_records[step - self.delay_step]
        else:
            return 0
    
    def ignite(self,total_time,time_step = 0.01,delay_time = 0.1):
        random_input_span = (1.2,2.8)
        
        total_steps = int(total_time/time_step)
        time_span = np.arange(0,total_time,time_step)
    
        self.delay_step = int(delay_time/time_step)
        
        theta_arr = np.random.uniform(0,2*np.pi,size = self.num_neurons)
        
        
        m_arr = np.zeros(total_steps)
        e_arr = np.zeros(total_steps)
        
        random_input = np.random.uniform(*random_input_span,size = self.num_neurons)
        
        self.amin_saman_param = np.zeros( total_steps )
        self.spiking_records = np.array( np.zeros(total_steps) )
        self.completed_laps = np.zeros( self.num_neurons )
        
        for i in tqdm( range(total_steps - 1),desc = 'network dynamic' ):
            theta_arr = theta_arr + (random_input - np.cos(theta_arr) - self.g * e_arr[i] )*time_step
            
            #here we should spot the spiking neurons.
            mask = theta_arr > np.pi
            self.spiking_records[i] = np.sum( mask )
            theta_arr = theta_arr - 2 * np.pi * mask
            
            m_arr[i+1] = m_arr[i] + time_step*( -self.alpha*m_arr[i] ) + ( (self.alpha**2)/self.num_neurons ) *self._retarded_spikes_record(i) 
            e_arr[i+1] = e_arr[i] + time_step*( m_arr[i] - self.alpha*e_arr[i] )
            
            self.amin_saman_param[i] = np.mean( np.sin(theta_arr) )**2
            
        
        
        self.total_time = total_time
        self.total_steps = total_steps
        self.time_step = time_step
        self.theta_arr = theta_arr
        self.effective_field_array = e_arr
        return
    
    def report_sync_parameter(self,last_steps_period = 1000):
        if self.total_steps < 1000: last_steps_period = self.total_steps
        return np.mean( self.amin_saman_param[-last_steps_period:] )
    
    def report_sigma(self):
        sigma = np.std( self.effective_field_array )
        return sigma
    
    
    pass

# sample_model = Rotational_neural_network(num_neurons=1000,g=0.1,alpha = 100)
# sample_model.ignite(total_time = 1000)
# sync_param = sample_model.report_sync_parameter()
# sigma = sample_model.report_sigma()