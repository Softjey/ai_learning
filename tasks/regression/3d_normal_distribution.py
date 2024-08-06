import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 

def random_distribution(N: int) -> tuple:
    x = np.random.rand(N) * 100
    y = np.random.rand(N) * 100
    z = np.random.rand(N) * 100

    return x, y, z
  
def normal_distribution(N: int) -> tuple:
    x = np.random.randn(N) * 100
    y = np.random.randn(N) * 100
    z = np.random.randn(N) * 100

    return x, y, z

figure = plt.figure()
ax = figure.add_subplot(121, projection='3d')
ax2 = figure.add_subplot(122, projection='3d')

N = 400
x, y, z = random_distribution(N)
x_n, y_n, z_n = normal_distribution(N)

x_m = 2.23
z_m = -1.53
b = -56

calculated_y = x_m * x + z_m * z + b 
calculated_y_n = x_m * x_n + z_m * z_n + b


ax.scatter(x, calculated_y, z, color='blue')
ax2.scatter(x_n, calculated_y_n, z_n, color='red')

plt.show()