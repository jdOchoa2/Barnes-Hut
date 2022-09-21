''' Animate the evolution of galaxy'''

from common import *
import time
from numpy import loadtxt


Data = loadtxt('Parameters', dtype=str)

# Number of bodies.
N = int(Data[2])

# Number of time-iterations executed by the program.
n = int(Data[7])

# Frequency at which data is written.
save_step = int(Data[8])

# Folder to save the images
image_folder = str(Data[9])

# Folder to save the data
data_folder = Data[10]

# Format of files
format = Data[11]

# Name of the generated video
video_name = str(N)+'bodies.mp4'

start = time.time()
read_evolution(N, n, save_step, data_folder, format, image_folder)
create_video(image_folder, video_name)
end = time.time()
total_time = end - start
print(f'\nAnimation spent {total_time:.2f} s')