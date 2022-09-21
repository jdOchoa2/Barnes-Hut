''' Evolution of a N body sistem interacting gravitationally'''

from common import *
import time
from numpy import loadtxt


Data = loadtxt('Parameters', dtype=str)

# Number of bodies.
N = int(Data[2])

# Initial radius of the distribution
ini_radius = float(Data[3]) #kpc

# Number of time-iterations executed by the program.
n = int(Data[7])

# Frequency at which data is written.
save_step = int(Data[8])

# Folder to save the images
image_folder = Data[9]

# Folder to save the data
data_folder = Data[10]

# Format of files
format = Data[11]

# Evolution 
start = time.time()
bodies = system_init_read(N, format, data_folder)
print(f'\nSystem is compounded of {len(bodies):.0f} bodies')
print(f'\nTotal mass of the system is: {sum(bodies[i].m for i in range(len(bodies))):.0f} Msun\n')
evolve(bodies, N, n, ini_radius, save_step, data_folder, format)
end = time.time()

total_time = end - start
print(f'\nEvolution spent {total_time:.2f} s')