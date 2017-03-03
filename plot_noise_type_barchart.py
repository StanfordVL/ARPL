import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", context="talk", font_scale=1.2)
rs = np.random.RandomState(7)

# Set up the matplotlib figure
fig, ax = plt.subplots()

# Generate some sequential data
x = np.array(["nominal", "random\ndynamics", "random\nprocess", "random\nobservation", "adversarial\ndynamics", "adversarial\nprocess", "adversarial\nobservation"])

y = np.array([2779.5, 1153.4, 1893.9, 1476.5, 1059.8, 448.6, 1703.5])

sns.barplot(x, y, palette="Set3", ax=ax)
ax.set_ylabel("Performance")

# # TODO: replace this with real data
# y_std = np.array([10, 10, 20, 20, 30, 30, 40]) * 10
# ax.errorbar(np.arange(7), y, yerr=[y_std, y_std], fmt='.', capthick=2)

# Finalize the plot
sns.despine(bottom=False, left=False)
plt.savefig('noise.pdf')