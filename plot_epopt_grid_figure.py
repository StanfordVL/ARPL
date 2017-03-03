from string import letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cPickle as pickle

sns.set(style="white", font_scale=2)

# Generate a large random dataset
# rs = np.random.RandomState(33)

grid = pickle.load(open('grid_rollouts_InvertedPendulumDynamic-v1'))
grid = np.array(grid)

# d = pd.DataFrame(data=grid.mean(axis=4).mean(axis=1)[0][::-1,:])
# d = pd.DataFrame(data=grid.mean(axis=4).mean(axis=1)[1][::-1,:])
d = pd.DataFrame(data=grid.mean(axis=4).mean(axis=1)[2][::-1,:])

# Compute the correlation matrix
# corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(d, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# mask = np.ones_like(corr, dtype=np.bool)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d, cmap=cmap, vmin=0, vmax=1000.0, square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

ax.set_xlabel("Mass of Cart", fontsize=24)
ax.set_ylabel("Mass of Pole", fontsize=24)
# ax.set_title('Inverted Pendulum (Nominal)', fontsize=24)
# ax.set_title('Inverted Pendulum (Random)')
ax.set_title('Inverted Pendulum (Adversarial)')

ax.set_xticklabels(['%d%%'%i for i in [50, 60, 70, 80, 90, 110, 120, 130, 140, 150]], fontsize=18)
ax.set_yticklabels(['%d%%'%i for i in [50, 60, 70, 80, 90, 110, 120, 130, 140, 150]], fontsize=18)

# plt.show()
# plt.savefig('epopt_nominal.pdf')
# plt.savefig('epopt_random.pdf')
plt.savefig('epopt_adversarial.pdf')