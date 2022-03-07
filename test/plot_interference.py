import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(1, 16)
y1 = [1-4.2*(i-1)/4000 for i in range(1, 16)]
y2 = [100*(1-np.prod(y1[:i])) for i in range(1, 16)]
tmp = int(4000/4.2)
y3 = [100*(1-math.factorial(tmp)/math.factorial(tmp-i)/(tmp**i)) for i in range(1, 16)]
import pdb; pdb.set_trace()
print(y2)
print(y3)
plt.plot(x, y2)
plt.grid()
plt.xlabel('Number of radars', fontsize=18)
plt.ylabel('Probability of interference (%)', fontsize=18)
plt.show()
