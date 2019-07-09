import matplotlib.pyplot as plt

fig = plt.subplot()

ls0, = fig.plot([2], [1], '.')
plt.pause(0.05)
plt.show()
