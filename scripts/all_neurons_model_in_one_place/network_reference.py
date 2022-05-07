neuron_engine = 'Non_repulsive_rotational'

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:02:33 2021

@author: mohsen
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

from tqdm.notebook import tqdm as tqdm
# tqdm().pandas() #This line make sure that the progress bars looks natural

from neurons_engines import Rotational_neural_network, IF_neural_network, Non_repulsive_rotational_neural_network

engines_dict = {'IF':IF_neural_network, 'Rotational':Rotational_neural_network, 'Non_repulsive_rotational':Non_repulsive_rotational_neural_network}
network_engine_class = engines_dict[neuron_engine]


class Network_of_neurons(network_engine_class):
    def __init__(self,num_neurons,g,alpha = 20):
        super().__init__(num_neurons,g,alpha)
        
        self.warp_num = 100 #vertical axis
        self.weft_num = 100 #horizental axix
        
        # self.potentail_arr = np.zeros(self.num_neurons,dtype = float)
        
        self.driving_wind = np.zeros(self.num_neurons,dtype = float)
        self.spike_mask = np.zeros(self.num_neurons,dtype = bool)
        
        
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
        sigma = np.std( self.e_arr[- self.total_steps//2:] )
        return sigma
    
    def report_e_period_fft(self):
        yf = np.abs( np.fft.fft( self.e_arr ) )[1:self.e_arr.size//2] #remove the constant variable
        xf = np.fft.fftfreq(self.e_arr.size, d = self.time_step)[1:self.e_arr.size//2] #we want only the first half containing the positive freqs.
        
        self.max_intensity = np.max(yf)
        # max_index = np.where( yf == self.max_intensity )[0][0] #we need the index not the array including it!
        max_index = np.argmax(yf)
        self.e_period = 1 / xf[max_index]
        return self.e_period, self.max_intensity

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


class Animated_network_of_neurons(Network_of_neurons):
    def __init__(self,neuron_model,num_neurons,g,alpha = 20,**kwargs):
        super().__init__(num_neurons,g,alpha)
        # self.potentail_arr = np.random.uniform(-np.pi,np.pi, size = num_neurons)
        
        current_models = ['IF','Rotational','Non_repulsive_rotational']
        
        
        self.warp_num = 100 #vertical axis
        self.weft_num = 100 #horizental axix
        
        # self.potentail_arr = np.zeros(self.num_neurons)
        # self.potentail_arr = np.tile( np.linspace(-np.pi,np.pi, num = int(num_neurons/self.weft_num) ), reps = self.weft_num )
        self.potentail_arr = np.random.uniform(-np.pi,np.pi, size = num_neurons)
        
        if neuron_engine == current_models[0]:
            # self.random_input_span = (1.2,2.8)
            self.random_input_span = kwargs.get('random_input_span', (1.2,2.8) )
            self.ceiling_state = 1
            self.floor_state =  - 0.5 #used for animations frame
            
            self.important_states_namestrings = [r'-0.5', '0', '0', str(self.ceiling_state)]
            
            self.wind_name = r'$\dot v$'
            self.wind_amplitude = [-1,1]
        
        elif neuron_engine in current_models[1:3]:
            self.random_input_span = kwargs.get('random_input_span', (9.5,13.5) )
            # self.random_input_span = (9.5,13.5)
            self.ceiling_state = np.pi
            self.floor_state = - 5*np.pi/2 #used for animations frame
            
            self.important_states_namestrings = [r'$-5\frac{\pi}{2}$', '$-\pi$', '0', '$\pi$']            
            
            self.wind_name = r'$\dot\theta$'
            self.wind_amplitude = [-13,13]
        
        self.ax_xlim = kwargs.get('xlim',[self.random_input_span[0],self.random_input_span[1]])
        self.ax_ylim = kwargs.get('ylim',[self.floor_state, self.ceiling_state])        
            
    def brace_for_lunch(self,total_time,time_step = 0.01,delay_time = 0.1):
        
        # random_input_span = (3.5,13.5)
        
        total_steps = int(total_time/time_step)
        time_span = np.arange(0,total_time,time_step)
        
        self.total_time = total_time
        self.total_steps = total_steps
        self.time_step = time_step
        self.delay_step = int(delay_time/time_step)
        
        
        self.m_arr = np.zeros(total_steps)
        self.e_arr = np.zeros(total_steps)
        # self.random_input = np.random.uniform(*self.random_input_span,size = self.num_neurons)
        
        # self.random_input = np.linspace(self.random_input_span[0], self.random_input_span[1], num=self.weft_num)
        # self.random_input = np.repeat(self.random_input, repeats = int( self.num_neurons / self.weft_num) )
        
        self.random_input = np.random.uniform(self.random_input_span[0], self.random_input_span[1], size = self.num_neurons)
        
        self.amin_saman_param = np.zeros( total_steps )
        self.spiking_records = np.zeros(total_steps)
        
        self._set_cornometers()
    
        return

    def _init_ax(self):
        
        ax = self.fig.add_subplot(self.gs[0,0])
        ax_stat = self.fig.add_subplot(self.gs[0,1], sharey = ax)
        
        ax.set_yticks([self.floor_state, - np.pi, 0, self.ceiling_state])
        ax.set_yticklabels(self.important_states_namestrings)
        ax.set_title('Network dynamic N={} g={}'.format(self.num_neurons,self.g))
        self._compute_suitable_extents()
        
        self.colored_plateau = ax.imshow( self.plateau, aspect= 'auto', extent = self.extent , vmin = 0, vmax = 10, cmap = 'hot')

        # ax.set_xlim(self.ax_xlim)
        ax.set_ylim(self.ax_ylim)
        # ax.invert_yaxis()
        
        ax_stat.set_xlabel('population')
        self.pop_dist, = ax_stat.plot( np.sum(self.plateau>0,axis = 1), np.linspace(self.floor_state,self.ceiling_state,num = self.warp_num) )
        ax_stat.set_xlim([0,self.num_neurons/self.warp_num])
        return
    
    
    def _init_ax_e(self):
        ax_e = self.fig.add_subplot(self.gs[2, 0])
        
        ax_e.set_ylabel('E')
        ax_e.set_xlabel('t')

        ax_e.set_xlim([0,1])
        ax_e.set_ylim([-0.5,2.5])

        time_series = np.arange(0,0.8,self.time_step)
        self.e_pulse, = ax_e.plot(time_series,time_series*0)

        return
    
    def _init_ax_theta_dot(self):
        
        ax_theta_dot = self.fig.add_subplot(self.gs[1, 0])
        ax_theta_dot.set_ylabel(self.wind_name)
        ax_theta_dot.set_xlabel('neuron inputs')
        self.wind_direction, = ax_theta_dot.plot(self.random_input[self.argsort_inputs], self.driving_wind[self.argsort_inputs],'ro',markersize=1)
        # self.wind_direction, = ax_theta_dot.plot(range(1,self.num_neurons+1), np.zeros(self.num_neurons))
        ax_theta_dot.set_ylim(self.wind_amplitude)
        
        return
        
    def _compute_figure_grid(self):
        if self.show_space == True:
            ncols = 2
        else:
            ncols = 1
        nrows = np.sum([self.show_space,  self.show_field, self.show_velocity])
        self.gs = gridspec.GridSpec(nrows = nrows, ncols = ncols, wspace = 0.2)
        # self.gs = gridspec.GridSpec(3, 2, width_ratios = (10,4), height_ratios = (10,5,2), wspace = 0.2)

        return self.gs
    
    def init(self):
        
        self._compute_figure_grid()
        if self.show_space == True: self._init_ax()
        if self.show_field == True: self._init_ax_e()
        if self.show_velocity == True: self._init_ax_theta_dot()
        return
    
    
    def update(self,frame):
        
        was_network_active = np.sum(self.spike_mask) > 0
        
        self.plateau[:]=self.plateau[:]*0
        self._march_on(frame)
        phase_marks = np.floor( (self.potentail_arr - self.floor_state) / self.grating_blocks_length ).astype('int') #sorted neurons
        
        # Change the spiking group color if they stopped spiking
        if np.sum(self.spike_mask) == 0 and was_network_active:
            # self.color_num += 2
            self.color_num = (self.color_num %10)
    
        # Update neuron phase marks and color
        for neuron_index in range(self.num_neurons):
            neuron_column = self.column_indices[neuron_index]
            neuron_mark = phase_marks[neuron_index]
            
            #Spiking ones
            if self.spike_mask[neuron_index] == True:
                self.color_marks[neuron_index] = self.color_num
                
            #coloring
            if neuron_mark >= 0 and neuron_mark <= self.warp_num:
                self.plateau[neuron_mark,neuron_column] = self.color_marks[neuron_index]
            
        
        self.colored_plateau.set_data(self.plateau)
        self.pop_dist.set_xdata( np.sum(self.plateau>0,axis = 1) )
        # self.e_pulse.set_ydata( np.roll( self.e_arr, shift= 80 - frame )[0:80] ) #takes care of initial steps
        # self.wind_direction.set_ydata(self.driving_wind[self.argsort_inputs])
        return self.plateau
    
    
    def _compute_column_indices(self):
        if self.random_input_span[1] != self.random_input_span[0]:
            indices = np.floor( (self.random_input - self.random_input_span[0])/(self.random_input_span[1] - self.random_input_span[0]) * (self.weft_num-1) ).astype('int')
        elif self.random_input_span[1] == self.random_input_span[0]:
            indices = np.repeat( np.arange(self.num_neurons / self.weft_num, dtype = int), repeats= self.weft_num)
        return indices
    
    
    def _compute_suitable_extents(self):
        if self.random_input_span[1] != self.random_input_span[0]:
            self.extent = [self.random_input_span[0] , self.random_input_span[1], self.ceiling_state , self.floor_state] #imshow axises are updside down    
        elif self.random_input_span[1] == self.random_input_span[0]:
            self.extent = [1 , self.weft_num, self.ceiling_state , self.floor_state] #imshow axises are updside down    
        return
    
    
    def render_animation(self, start_time, show_space = True,
                         show_field = False, show_velocity = False, path = None):
        self.fig = plt.figure()
        plt.rc('font', family='serif')
        plt.style.use('dark_background')

        self.column_indices = self._compute_column_indices()
        
        self.argsort_inputs = np.argsort(self.column_indices)
        
        self.grating_blocks_length = ( self.ceiling_state - self.floor_state )/self.warp_num
        self.plateau = np.zeros((self.warp_num, self.weft_num))
        
        self.color_num = 5
        self.color_marks = np.ones(self.num_neurons) * self.color_num
        
        self.show_space = show_space
        self.show_field = show_field
        self.show_velocity = show_velocity
        

        # self.fig.tight_layout()
        
        self.init()
        frames_range = range( int(start_time/self.time_step), self.total_steps)
        # ani = FuncAnimation(self.fig, self.update, init_func = self.init, frames= frames_range, interval = 50)
        self.ani = FuncAnimation(self.fig, self.update, init_func = None, frames= frames_range, interval = 50)
        # if path != None: ani.save(path, writer='imagemagick')

        return
    pass

    
# sample_model = Network_of_neurons(num_neurons= 10000, g = 15)
# sample_model.ignite(random_input_span = (9.5,13.5), total_time = 1000, delay_time = 0.1)

# sigma = sample_model.report_sigma()
# e_period = sample_model.report_e_period()
# e_mean = sample_model.e_mean

# spikies_period_mean = sample_model.report_spikies_period_mean()