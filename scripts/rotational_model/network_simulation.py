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
        
        random_input_span = (3.5,13.5)
        
        total_steps = int(total_time/time_step)
        time_span = np.arange(0,total_time,time_step)
        
        self.total_time = total_time
        self.total_steps = total_steps
        self.time_step = time_step
    
        self.delay_step = int(delay_time/time_step)
        
        self.theta_arr = np.zeros(self.num_neurons,dtype = float)
        
        m_arr = np.zeros(total_steps)
        e_arr = np.zeros(total_steps)
        
        random_input = np.random.uniform(*random_input_span,size = self.num_neurons)
        
        self._set_cornometers()
        
        self.amin_saman_param = np.zeros( total_steps )
        self.spiking_records = np.array( np.zeros(total_steps) )
        
        for i in tqdm( range(total_steps - 1),desc = 'network dynamic' ):
            #potential to fall
            # potential_free_fall_mask = self.theta_arr > - np.pi/2
            self.theta_arr = self.theta_arr + (random_input - self.g * e_arr[i] )*time_step
            
            #here we should spot the spiking neurons.
            self.spike_mask = self.theta_arr > np.pi
            self.spiking_records[i] = np.sum( self.spike_mask )
            self.theta_arr = self.theta_arr - 2 * np.pi * self.spike_mask
            
            # Prevent neurons from free falling to large negative degrees.
            # free_fall_mask = self.theta_arr < - np.pi/2
            # self.theta_arr[free_fall_mask & potential_free_fall_mask] = - np.pi/2
            
            m_arr[i+1] = m_arr[i] + time_step*( -self.alpha*m_arr[i] ) + ( (self.alpha**2)/self.num_neurons ) *self._retarded_spikes_record(i) 
            e_arr[i+1] = e_arr[i] + time_step*( m_arr[i] - self.alpha*e_arr[i] )
            
            self._record_amin_saman_param(i)
            self._update_cornometers()
            

        self.effective_field_array = e_arr
        return
    
    def _record_amin_saman_param(self,step):
        left_handed_neurons = np.cos(self.theta_arr) < 0
        self.amin_saman_param[step] = np.mean( np.sin(self.theta_arr[left_handed_neurons]) )**2
        return
    
    def _set_cornometers(self):
        """
        These cornometers are going to record mean spikes intervals for every neuron.

        """
        self.spiking_cornometer = np.zeros(self.num_neurons)
        self.total_spikes_intervals = np.zeros(self.num_neurons)
        self.total_spikes_num = np.zeros(self.num_neurons)
        return
    
    def _update_cornometers(self):
        """
        Set lapsed time for every neuron which spiked. In addition, let other
        cornometers to keep on ticking.

        """
        self.total_spikes_num = self.total_spikes_num + self.spike_mask
        self.total_spikes_intervals = self.total_spikes_intervals + self.spiking_cornometer * self.spike_mask #add to spike lapsed time
        self.spiking_cornometer = self.spiking_cornometer * (self.spike_mask-1)*(-1) #reset the spikied cornos
        self.spiking_cornometer = self.spiking_cornometer + (self.spike_mask-1)*(-1) #update the not spikied ones
        return
    
    def report_sync_parameter(self,last_steps_period = 1000):
        if self.total_steps < 1000: last_steps_period = self.total_steps
        return np.mean( self.amin_saman_param[-last_steps_period:] )
    
    def report_sigma(self):
        sigma = np.std( self.effective_field_array )
        return sigma
    
    def report_spikies_period_mean(self):
        """
        Wrap up the results recorded by cornometers.

        Returns
        -------
        spikies_period_mean
            an array of total neurons number which shows the mean period of spiking.

        """
        self.spikies_period_mean = self.total_spikes_intervals * self.time_step / self.total_spikes_num
        return self.spikies_period_mean
    
    
    pass

# sample_model = Rotational_neural_network(num_neurons=10000,g=30,alpha = 20)
# sample_model.ignite(total_time = 1000)
# sync_param = sample_model.report_sync_parameter()
# sigma = sample_model.report_sigma()
# neurons_mean_period = sample_model.report_spikies_period_mean()