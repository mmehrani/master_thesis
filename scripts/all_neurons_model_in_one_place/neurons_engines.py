# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:07:00 2021

@author: mohsen
"""

import numpy as np

class Rotational_neural_network:
    def __init__(self,num_neurons,g,alpha = 20, external_input = 0):
        self.num_neurons = num_neurons
        self.g = g
        self.alpha = alpha
        self.potentail_arr = np.random.uniform(-np.pi,np.pi, size = num_neurons)
        self.external_input = external_input
        return
    
    def _march_on(self,i):
        """
        Push the network forward in time.

        Parameters
        ----------
        i : integer
            This is the steps number during the dynamic

        Returns
        -------
        None.

        """

        #dynamics
        try:
            iter_object = iter(self.external_input)
            self.driving_wind = self.random_input - np.cos(self.potentail_arr) - self.g * self.e_arr[i-1] + self.external_input[i-1]
        except:
            self.driving_wind = self.random_input - np.cos(self.potentail_arr) - self.g * self.e_arr[i-1] + self.external_input
        
        self.potentail_arr = self.potentail_arr + self.driving_wind * self.time_step

        self.m_arr[i] = self.m_arr[i-1] + self.time_step*( -self.alpha*self.m_arr[i-1] ) + ( (self.alpha**2)/self.num_neurons ) *self._retarded_spikes_record(i-1) 
        self.e_arr[i] = self.e_arr[i-1] + self.time_step*( self.m_arr[i-1] - self.alpha*self.e_arr[i-1] )
        
        #here we should spot the spiking neurons.
        self.spike_mask = self.potentail_arr > np.pi
        self.spiking_records[i] = np.sum( self.spike_mask )
        self.potentail_arr = self.potentail_arr - 2 * np.pi * self.spike_mask
        
    
        self._record_amin_saman_param(i)

        return
    
    def _record_amin_saman_param(self,step):
        left_handed_neurons = np.cos(self.potentail_arr) < 0
        self.amin_saman_param[step] = np.mean( np.sin(self.potentail_arr[left_handed_neurons]) )**2
        return
    
    
    def report_sync_parameter(self,last_steps_period = 1000):
        if self.total_steps < 1000: last_steps_period = self.total_steps
        return np.mean( self.amin_saman_param[-last_steps_period:] )

    
    pass

class Non_repulsive_rotational_neural_network(Rotational_neural_network):
    def __init__(self,num_neurons,g,alpha = 20, external_input = 0):
        self.num_neurons = num_neurons
        self.g = g
        self.alpha = alpha
        self.potentail_arr = np.random.uniform(-np.pi,np.pi, size = num_neurons)
        self.external_input = external_input
        return
    
    def _march_on(self,i):
        """
        Push the network forward in time.

        Parameters
        ----------
        i : integer
            This is the steps number during the dynamic

        Returns
        -------
        None.

        """

        #dynamics
        try:
            iter_object = iter(self.external_input)
            if len(self.e_arr) == len(self.external_input):
                self.driving_wind = self.random_input - self.g * self.e_arr[i-1] + self.external_input[i-1]
            else:
                raise AttributeError('External input does not have correct length!')
            
        except:
            self.driving_wind = self.random_input - self.g * self.e_arr[i-1] + self.external_input
            
        self.potentail_arr = self.potentail_arr + self.driving_wind * self.time_step

        self.m_arr[i] = self.m_arr[i-1] + self.time_step*( -self.alpha*self.m_arr[i-1] ) + ( (self.alpha**2)/self.num_neurons ) *self._retarded_spikes_record(i-1) 
        self.e_arr[i] = self.e_arr[i-1] + self.time_step*( self.m_arr[i-1] - self.alpha*self.e_arr[i-1] )
        
        #here we should spot the spiking neurons.
        self.spike_mask = self.potentail_arr > np.pi
        self.spiking_records[i] = np.sum( self.spike_mask )
        self.potentail_arr = self.potentail_arr - 2 * np.pi * self.spike_mask
        
    
        self._record_amin_saman_param(i)

        return

class IF_neural_network:
    def __init__(self,num_neurons,g,alpha = 20, external_input = 0):
        self.num_neurons = num_neurons
        self.g = g
        self.alpha = alpha
        self.potentail_arr = np.random.uniform(0,1, size = num_neurons)
        self.external_input = external_input
        return

    def _march_on(self,i):
        """
        Push the network forward in time.

        Parameters
        ----------
        i : integer
            This is the steps number during the dynamic

        Returns
        -------
        None.

        """
        try:
            iter_object = iter(self.external_input)
            self.driving_wind = self.random_input - self.potentail_arr - self.g * self.e_arr[i-1] + self.external_input[i-1]
        except:
            self.driving_wind = self.random_input - self.potentail_arr - self.g * self.e_arr[i-1] + self.external_input
            
        self.potentail_arr = self.potentail_arr + self.driving_wind *self.time_step 
        
        #here we should spot the spiking neurons.
        self.spike_mask = self.potentail_arr > 1
        self.spiking_records[i] = np.sum( self.spike_mask )
        self.potentail_arr = self.potentail_arr - 1*(self.potentail_arr > 1)
        
        self.m_arr[i] = self.m_arr[i-1] + self.time_step*( -self.alpha*self.m_arr[i-1] ) + ( (self.alpha**2)/self.num_neurons ) *self._retarded_spikes_record(i-1) 
        self.e_arr[i] = self.e_arr[i-1] + self.time_step*( self.m_arr[i-1] - self.alpha*self.e_arr[i-1] )
        
        
        return
        
    pass