#!/usr/bin/env python
# coding: utf-8

# # Neural networks simulation (Synchronization Problem)

# This file is going to study any neural netwrok class which is defined in the `<network_reference.py>` file.

# In[1]:


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# In[2]:


get_ipython().run_cell_magic('capture', '', 'from tqdm import tqdm_notebook as tqdm\ntqdm().pandas() #This line make sure that the progress bars looks natural')


# ## Order your neuron model of interest

# In[3]:


current_models = ['IF','Rotational','Non_repulsive_rotational']
neuron_model = current_models[2]

with open("network_reference.py") as net_ref: 
    lines = net_ref.readlines() #read 
 
    #modify 
    lines[0] = "neuron_engine = '{}'\n".format(neuron_model) #you can replace zero with any line number. 
 
with open("network_reference.py", "w") as net_ref: 
    net_ref.writelines(lines) #write back 


# In[4]:


from network_reference import Network_of_neurons


# ## Parameters and constants

# In[5]:


num_neurons = 10000
total_time = 100

num_ensembles_each_g = 1


# In[6]:


# random_input_span = (9.5,13.5)
random_input_span = (9.5,9.5)
# random_input_span = (1.2,2.8)


# In[7]:


connectivity_min,connectivity_max, connectivity_step = 0, 20, 0.5
connectivity_span = np.arange(connectivity_min,connectivity_max,connectivity_step).round(2)


# In[8]:


delay_min, delay_max, delay_step =0.1, 0.11, 0.02
delay_span = np.arange(delay_min, delay_max, delay_step).round(2)


# In[9]:


alpha_min, alpha_max, alpha_step = 20, 25, 10
alpha_span = np.arange(alpha_min, alpha_max, alpha_step).round(2)


# In[ ]:





# ## Prepare the storage to record the outputs

# In[10]:


version = '_similar_input'
current_size_networks_path = os.path.join(neuron_model + '_ensembles','N{}_T{}_I{}_{}'.format(num_neurons,total_time,random_input_span[0],random_input_span[1]) + version )
try:
    os.makedirs(current_size_networks_path)
except:
    pass


# In[11]:


from storage_modifications import *


# ## Creating various ensembles:

# In[ ]:


for connectivity in tqdm( connectivity_span , desc='Variation of connectivities'):
    group_folder_name = 'g_{}'.format(connectivity)
    group_storage = make_inner_dir(current_size_networks_path, group_folder_name)
    
    for delay in tqdm( delay_span , desc='Variation of delay'):
        d_subgroup_folder_name = 'd_{}'.format(delay)
        d_subgroup_storage = make_inner_dir(group_storage, d_subgroup_folder_name)
        
        for alpha in tqdm( alpha_span , desc='Variation of alpha') :
            alpha_subgroup_folder_name = 'alpha_{}'.format(alpha)
            alpha_subgroup_storage = make_inner_dir(d_subgroup_storage, alpha_subgroup_folder_name)
            last_index = last_ensemble_index_in_group(alpha_subgroup_storage)

            for i in tqdm( range(num_ensembles_each_g) , desc='Making ensembles'):
                sample_model = Network_of_neurons(num_neurons=num_neurons,
                                                  g=connectivity, alpha = alpha)
                sample_model.ignite(random_input_span, total_time = total_time,
                                    time_step = 0.01, delay_time = delay)

                current_ensemble_folder = make_inner_dir(alpha_subgroup_storage, str(last_index + i + 1) )

                save_sigma(sample_model.report_sigma(), current_ensemble_folder)
                save_field_period(*sample_model.report_e_period_fft(), current_ensemble_folder)
                save_field_average(sample_model.e_arr, current_ensemble_folder)
                
                if neuron_model in current_models[1:]: save_amin_saman_param(sample_model.report_sync_parameter(), current_ensemble_folder)
                save_mean_spiking_periods( sample_model.report_spikies_period_mean(), current_ensemble_folder )


# In[ ]:





# In[ ]:


get_ipython().system('ipython nbconvert --to python any_neural_model_ensembles.ipynb')


# In[ ]:




