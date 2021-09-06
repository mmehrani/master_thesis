neuron_engine = 'Rotational'

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:02:33 2021

@author: mohsen
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from neurons_engines import Rotational_neural_network, Kuramoto_neural_network

engines_dict = {'IF':Kuramoto_neural_network, 'Rotational':Rotational_neural_network}
network_engine_class = engines_dict[neuron_engine]


class Network_of_neurons(network_engine_class):
    def __init__(self,num_neurons,g,alpha = 20):
        self.num_neurons = num_neurons
        self.g = g
        self.alpha = alpha
        self.potentail_arr = np.zeros(self.num_neurons,dtype = float)
        return
    
    def _retarded_spikes_record(self,step):
        if step > self.delay_step: 
            return self.spiking_records[step - self.delay_step]
        else:
            return 0
    
    def ignite(self,random_input_span,total_time,time_step = 0.01,delay_time = 0.1):
        
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
        
        for i in tqdm( range(total_steps - 1),desc = 'network dynamic' ):
            self._march_on(i)
            self._update_cornometers()
            
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


    def report_sigma(self):
        sigma = np.std( self.e_arr )
        return sigma
    
    def report_e_period_fft(self):
        yf = np.abs( np.fft.fft( self.e_arr - np.mean(self.e_arr) ) ) #remove the constant variable
        xf = np.fft.fftfreq(self.e_arr.size, d = self.time_step)
        
        max_index = np.where( yf == np.max(yf) )[0][0] #we need the index not the array including it!
        self.e_period = 1 / xf[max_index]
        return self.e_period

    def plot_e_fft(self):
        yf = np.fft.fft( self.e_arr )
        xf = np.fft.fftfreq(self.e_arr.size, d = self.time_step)
        plt.plot(xf,yf)
        return
    
    def report_e_period(self, **kwargs):
        sampling_period = kwargs.get('sampling_period',int( self.total_steps / 10 ) )
        e_sampled = self.e_arr[-sampling_period:]
        self.e_mean = np.mean(e_sampled)
        
        e_booled = 1*(e_sampled > self.e_mean)
        
        #computing E(t) main period
        period_list = []
        start_index = 0
        index = 0
        for i in range(5):
            while e_booled[index] == e_booled[start_index]: # Find when it start to get a different value
                index += 1 
            
            period = 0
            try:
                while e_booled[index + period] != e_booled[0]: # How long this value resist
                    period += 1
                period*=2  # period must show the total oscillation time
                period_list.append(period)
                start_index = index
                
            except:
                if i == 0 : period_list.append(sampling_period)
                # period = sampling_period
                break
            
        self.e_period = round( np.mean(period_list) * self.time_step, 2)
        return self.e_period 

    pass



    
# sample_model = Network_of_neurons(num_neurons= 10000, g = 15)
# sample_model.ignite(random_input_span = (9.5,13.5), total_time = 1000, delay_time = 0.1)

# sigma = sample_model.report_sigma()
# e_period = sample_model.report_e_period()
# e_mean = sample_model.e_mean

# spikies_period_mean = sample_model.report_spikies_period_mean()