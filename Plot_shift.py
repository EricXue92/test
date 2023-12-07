import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 18})

#plt.rcParams['text.usetex'] = True

# sns.set(style="white", font_scale = 2)





# Generating data before the distrib__init__ution shift
np.random.seed(42)
before_shift = np.random.normal(loc= 30, scale=5, size=2000)

# Simulating a shift in distribution
after_shift = np.random.normal(loc= 40, scale= 6, size=2000)

# Simulating a OOD in distribution
# OOD = np.random.normal(loc= 70, scale= 4, size=100)

OOD = np.random.exponential(scale= 4, size=2000) + 60

# Plotting histograms to visualize the distribution shift
plt.figure(figsize=(8, 5), tight_layout=True)

sns.kdeplot(before_shift, color='black', label='Training data', lw = 3)
sns.kdeplot(after_shift, color='blue', label='Distributional shift', lw = 3)

sns.kdeplot(OOD, color='red', label='OOD', lw = 3)

# plt.hist(before_shift, bins=30, alpha=0.4, color='blue', label='Training data')
# plt.hist(after_shift, bins=30, alpha=0.4, color='red', label='Shift data')
plt.xlabel('x')
plt.grid(linestyle='dotted')
plt.legend()
plt.tight_layout() 
plt.savefig('Shift.pdf')
plt.show()