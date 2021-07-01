# Neurons in sync
## (Markovian approch toward modeling networks of spiking neurons)

In this project we have studied how the strength of connectivity between neurons pairs will control the phase transition.

## Dynamics of the neurons

$$
\begin{cases}
\dot{\theta_i}=I_i - cos(\theta_i) -g  E \\
\ddot{E}+ 2\alpha \dot{E}+\alpha^{2}E =\frac{\alpha^{2}}{N} \sum_{n|tـn<t} \delta(t - t_n - t_d)
\end{cases}
$$
We have used mean-field approximation in order of simulation. In other words, the network is fully connected and each neuron is free to influence all the present neurons or even get influenced by so.

## Synchronized phase
The following animation shows how the total population of neurons got synchronized by choice of corresponding value of connecitivity.
![](..\scripts\all_neurons_model_in_one_place\animations\sea_shore\well_in_negatives\N10000_g20_Imin9.5_Imax13.5_neurons_rotational.gif)

## Insynchronized phase
When the conncetivity is low, neurons walks around almost ignoring the other exisiting ones.
![](..\scripts\all_neurons_model_in_one_place\animations\sea_shore\well_in_negatives\N10000_g10_Imin9.5_Imax13.5_neurons_rotational.gif)

## Source codes
you may have your own produced animation from this [python code]().
## Welcome to GitHub Pages



```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```


### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mmehrani/master_thesis/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
