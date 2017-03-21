# Adversarially Robust Policy Learning through Active construction of Physically-plausible Perturbations

<iframe width="560" height="315" src="https://youtu.be/yZ-gSsbbzh0" frameborder="10" allowfullscreen></iframe>
\[Placeholder video -- deadline Mar 5\]

### Abstract
Policy learning in reinforcement learning has demonstrated success in scaling up problem size beyond toy examples. However, enabling these methods on real robots poses a challenge of both sample complexity during learning and safety against malicious intervention. Model-based method using simulated approximations of the target domains offer a possible solution, with the caveat that algorithms need to adapt across errors in modelling and adversarial perturbations. We introduce Adversarially Robust Policy Learning (ARPL) that leverage active construction of physically-plausible adversarial examples during training to enable sample-effiecient policy learning in source and resulting in a robust policy that performs well under both random perturbations as well as adversarial input manipulations. We further show that ARPL improves distance to uncontrollability in simplified environments, hence providing a justification for improved robustness in more complex environments. We evaluate our method on four tasks with continuous control and show superior performance of ARPL as compared to state-of-the-art robust policy learning methods.

### Methods
We work with a physical dynamical system model:
$$
x_{t+1} = f(x_t, u_t; \mu) + \nu
$$

$$
z_t = g(x_t) + \omega
$$

Here, $x_t$ denotes the state of the system at time $t$, $u_t$ denotes the control, or action, applied at time $t$, $f$ models the transition function (dynamics) of the system, $\mu$ parametrizes $f$, $\nu$ models process noise, $g$ models the observation function of the system, and $\omega$ denotes observation noise. We introduce adversarial perturbations to 3 key components of this system - perturbations in the dynamics ($\mu$), the process ($\nu$), and the observation ($\omega$). 

We refer to these respectively as **dynamics noise**, **process noise**, and **observation noise**. Dynamics noise refers to a perturbation in underlying system parameters such as mass or friction, while process noise and observation noise relate to direct perturbation of the state and observation. In order to produce these perturbations, we use a scaled version of the gradient of the loss function with respect to the state that the agent observes. This is easily achieved via backpropagation in the policy network.

We investigate the effects of these 3 different types of noise in 4 Mujoco environments under different frequencies of perturbation during both train time and test time, and under both random and adversarial generation.

### Results

![fig1](figures/fig1.png)

![fig2](figures//fig2.png)

![fig3](figures/fig3.png)

![fig4](figures//fig4.png)

![table2](figures/table2.png)

### References
- [**Adversarially Robust Policy Learning through Active construction of Physically-plausible Perturbations**]().  
  Ajay Mandlekar\*, Yuke Zhu\*, Animesh Garg*, Li Fei-Fei, Silvio Savarese (\* denotes equal contribution).  
  *Under review at IEEE International Conference on Intelligent Robotics and Systems, (IROS) 2017*

### Authors and Contributors  

Ajay Mandlekar, [Yuke Zhu](https://web.stanford.edu/~yukez/), [Animesh Garg](http://ai.stanford.edu/~garg/)  
PIs: [Fei-Fei Li](http://vision.stanford.edu/feifeili/), [Silvio Savarese](cvgl.stanford.edu/silvio/)

### Support or Contact

Please Contact [Animesh Garg](http://ai.stanford.edu/~garg/) at [garg@cs.stanford.edu](mail:garg@cs.stanford.edu)
