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
y_std = np.array([736.4, 1136, 1466.8, 1324.4, 1074.4])

sns.barplot(x, y, palette="Set3", ax=ax)
ax.set_ylabel("Performance")

# Finalize the plot
sns.despine(bottom=False, left=False)
plt.savefig('noise.pdf')