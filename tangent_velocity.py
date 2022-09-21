'''Plot of tangent velocity of bodies vs radius'''

from common import *
import time
from numpy import loadtxt, float_


Data = loadtxt('Parameters', dtype=str)

# Number of bodies.
N = int(Data[2])

# Number of time-iterations executed by the program.
n = int(Data[7])

# Initial radius of the distribution
ini_radius = float(Data[3]) #kpc

# Frequency at which data is written.
save_step = int(Data[8])

# Inclination
i = float(Data[4])

# Longitud of ascending node
Omega = float(Data[5])

# Folder to save the data
data_folder = Data[10]

# Video Name
video_name = f'{N}-bodies_tv.mp4'

# Format of files
format = Data[11]

# Folder to save the images
image_folder = str(Data[12])

start = time.time()
tangent_velocity_distribution(N, n, ini_radius, save_step, Omega, i, data_folder, format, image_folder)
create_video(image_folder, video_name)
end = time.time()
total_time = end - start
print(f'\nComputation of tangent velocity spent {total_time:.2f} s')
