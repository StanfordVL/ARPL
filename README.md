# Adversarially Robust Policy Learning (ARPL)

This repository contains code that was used to generate results for our paper on Adversarially Robust Policy Learning (ARPL). It is an extension on TRPO (https://arxiv.org/abs/1502.05477) that uses actively-chosen adversarial examples in order to improve policy robustness to changes in environment states and dynamics. Note that this repository is **experimental** and only meant for research purposes. 



### Installation

1. Install rllab (https://github.com/openai/rllab) and gym (https://github.com/openai/gym).
2. Replace the installations with the ones contained in this repository.



### Usage

- See **full_pipeline.py** to see an example of the full training and evaluation pipeline. 
- See **train_trpo_curriculum.py** to train an agent. You can see some example configurations in **curriculum_config.py**.
- See **eval_trpo_phi.py** to see how to evaluate an agent. 