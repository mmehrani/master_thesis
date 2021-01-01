# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Kuramoto_neural_network:
    
    def __init__(self,num_neurons,g):
        self.num_neurons = num_neurons
        self.g = g
    
        return
    
    def _retarded_spikes_record(self,step):
        
        if step > self.delay_step: 
            return np.sum(self.spiking_records[step - self.delay_step])
        else:
            return 0
    
    def ignite(self,total_time,time_step = 0.01,delay_time = 0.1):
        random_input_span = (1.2,2.8)
        
        total_steps = int(total_time/time_step)
        time_span = np.arange(0,total_time,time_step)
    
        self.delay_step = int(delay_time/time_step)
        
        potentail_arr = np.zeros(self.num_neurons,dtype = float)
        random_input = np.random.uniform(*random_input_span,size = self.num_neurons)
        
        self.spiking_records = np.array( np.zeros((total_steps,self.num_neurons)), dtype = bool )
        
        for i in tqdm(range(total_steps),desc = 'network dynamic'):
            potentail_arr = potentail_arr + (random_input - potentail_arr)*time_step \
                                - (self.g / self.num_neurons )*self._retarded_spikes_record(i)
            self.spiking_records[i] = potentail_arr > 1
            reset_func = lambda v: v-1 if v>1 else v
            vfunc = np.vectorize(reset_func)
            potentail_arr = vfunc(potentail_arr)
        
        self.total_time = total_time
        self.total_steps = total_steps
        self.time_step = time_step
        self.potentail_arr = potentail_arr
        return
    
    def compute_effective_field(self,alpha = 20):
        
        m_arr = np.zeros(self.total_steps)
        e_arr = np.zeros(self.total_steps)
        
        for i in range(0,self.total_steps-1):
            m_arr[i+1] = m_arr[i] + self.time_step* \
                        ( -alpha*m_arr[i] ) + (alpha**2 / self.num_neurons ) * self._retarded_spikes_record(i)
            e_arr[i+1] = e_arr[i] + self.time_step*( m_arr[i] - alpha*e_arr[i] )
        
        self.effective_field_array = e_arr
        return
    
    def report_sigma(self):
        sigma = np.std( self.effective_field_array )
        return sigma
    
    
    pass

# sample_model = Kuramoto_neural_network(num_neurons=100,g=5)
# sample_model.ignite(total_time = 1000)
# sample_model.compute_effective_field( alpha = 20)
# sample_model.report_sigma()