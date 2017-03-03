import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle

name = 'HalfCheetah'

sns.set(font_scale=1.5)

# create a group of time series
# num_samples = 90
# group_size = 10
# x = np.linspace(0, 10, num_samples)
# group = np.sin(x) + np.linspace(0, 2, num_samples) + np.random.rand(group_size, num_samples) + np.random.randn(group_size, 1)
# df = pd.DataFrame(group.T, index=range(0,num_samples))
#
fig, ax = plt.subplots()
#
# # plot time series with seaborn
# sns.tsplot(data=df.T.values, ax=ax, condition='label1') #, err_style="unit_traces")

rewards = pickle.load(open('global_rewards_%sDynamic-v1'%name))
rewards = np.array(rewards)

mean = rewards.mean(axis=1).T
# std = rewards.std(axis=1)

# print mean.shape

names = ['nominal', 'dynamics random', 'dynamics adv.', 'process random', 'process adv.', 'obs. random', 'obs. adv.']

df = []
k = 0
for i in [0, 1, 3, 5, 2, 4, 6]:
  for j in range(500):
    if j % 20 == 0:
      df.append([names[i], j, mean[j, i]])
      k += 1

df = pd.DataFrame(df)
df.columns = ['exp', 'iter', 'reward']
print df

sns.tsplot(data=df, ax=ax, time='iter', unit='exp', condition='exp', value='reward', legend=True)

ax.set_ylabel("Reward")
ax.set_xlabel("Training iteration")
ax.set_title(name)

plt.legend(loc=2)
# plt.show()
plt.savefig('learning_%s.pdf'%name.lower())