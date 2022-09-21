# 3D Barnes-Hut Algorithm for evolution of a galaxy 

from numpy import array, empty, random, float, sqrt, exp, pi, sin, cos, tan, arctan, zeros, save, load, cross, dot
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from numpy.linalg import norm
from copy import deepcopy
from tqdm import tqdm

##### Simulation Parameters ###############################

# Gravitational constant in units of kpc^3 M_sun^-1 Gyr^-2
G = 4.4985022e-6

# Discrete time step.
dt = 1.e-2 #Gyr

# Theta-criterion of Barnes-Hut algorithm.
theta = 0.3

###########################################################

class Node:
    '''------------------------------------------------------------------------
    A node object will represent a body (if node.child is None) or an abstract
    node of the octant-tree if it has node.child attributes.
    ------------------------------------------------------------------------'''
    def __init__(self, m, position, momentum):
        '''----------------------------------------------------------
        Creates a child-less node using the arguments
        -------------------------------------------------------------
        .mass     : scalar m
        .position : NumPy array  with the coordinates [x,y,z]
        .momentum : NumPy array  with the components [px,py,pz]
        ----------------------------------------------------------'''
        self.m = m
        self.m_pos = m * position
        self.momentum = momentum
        self.child = None

    def position(self):
        '''----------------------------------------------------------
        Returns the physical coordinates of the node.
        ----------------------------------------------------------'''
        return self.m_pos / self.m
        
    def reset_location(self):
        '''----------------------------------------------------------
        Resets the position of the node to the 0th-order octant.
        The size of the octant is reset to the value 1.0
        ----------------------------------------------------------'''
        self.size = 1.0
        # The relative position inside the 0th-order quadrat is equal
        # to the current physical position.
        self.relative_position = self.position().copy()
        
    def place_into_octant(self):
        '''----------------------------------------------------------
        Places the node into next order octant.
        Returns the octant number according to the labels defined in 
        the documentation.
        ----------------------------------------------------------'''
        # The next order quadrant will have half the side of the current octant
        self.size = 0.5 * self.size
        return self.subdivide(2) + 2*self.subdivide(1) + 4*self.subdivide(0)

    def subdivide(self, i):
        '''----------------------------------------------------------
        Places the node into the next order octant along the direction
        i and recalculates the relative_position of the node inside 
        this octant.
        ----------------------------------------------------------'''
        self.relative_position[i] *= 2.0
        octant = 0
        if self.relative_position[i] >= 1.0:
            octant = 1
            self.relative_position[i] -= 1.0
        return octant    

def add(body, node):
    '''--------------------------------------------------------------
    Defines the octo-tree by introducing a body and locating it 
    according to three conditions (see documentation for details).
    Returns the updated node containing the body.
    --------------------------------------------------------------'''
    smallest_octant = 1.e-4 # Lower limit for the side-size of the octants
    
    # Case 1. If node does not contain a body, the body is put in here
    new_node = body if node is None else None
    
    if node is not None and node.size > smallest_octant:
        # Case 3. If node is an external node, then the new body can not
        # be put in there. We have to verify if it has .child attribute
        if node.child is None:
            new_node = deepcopy(node)
            # Subdivide the node creating 4 children
            new_node.child = [None for i in range(8)]
            # Place the body in the appropiate octant
            octant = node.place_into_octant()
            new_node.child[octant] = node
        # Case 2. If node is an internal node, it already has .child attribute
        else:
            new_node = node

        # For cases 2 and 3, it is needed to update the mass and the position
        # of the node
        new_node.m += body.m
        new_node.m_pos += body.m_pos
        # Add the new body into the appropriate octant.
        octant = body.place_into_octant()
        new_node.child[octant] = add(body, new_node.child[octant])
    return new_node

def gravitational_force(node1, node2):
    '''--------------------------------------------------------------
    Returns the gravitational force that node1 exerts on node2.
    A short distance cutoff is introduced in order to avoid numerical
    divergences in the gravitational force.
    --------------------------------------------------------------'''
    cutoff_dist = 1.e-4
    d12 =  node1.position() - node2.position()
    d = norm(d12)
    if d < cutoff_dist:
        # Returns no Force to prevent divergences!
        return array([0., 0., 0.])
    else:
        # Gravitational force
        return G*node1.m*node2.m*(d12)/d**3

def force_on(body, node, theta):
    '''--------------------------------------------------------------
    Barnes-Hut algorithm: usage of the octo-tree. force_on computes 
    the net force on a body exerted by all bodies in node "node".
    Note how the code is shorter and more expressive than the 
    human-language description of the algorithm.
    --------------------------------------------------------------'''
    # 1. If the current node is an external node,
    #    calculate the force exerted by the current node on b.
    if node.child is None or node.child == [None for ii in range(8)]:
        return gravitational_force(node,body)

    # 2. Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal
    #    node as a single body, and calculate the force it exerts on body b.
    if node.size <norm(node.position() - body.position())*theta:
        return gravitational_force(node,body)

    # 3. Otherwise, run the procedure recursively on each child.
    return sum(force_on(body, c, theta) for c in node.child if c is not None)

def verlet(bodies, root, theta, dt):
    '''--------------------------------------------------------------
    Velocity-Verlet method for time evolution.
    --------------------------------------------------------------'''
    for body in bodies:
        force = force_on(body, root, theta)
        body.momentum += 0.5*force*dt
        body.m_pos += body.momentum*dt
        body.momentum += 0.5*force_on(body, root, theta)*dt
    
def func(x,Distribution,Point): 
    """--------------------------------------------------------------
    Equation that follows the point of a wanted distribution that 
    matches the random one of a uniform distribution
    -----------------------------------------------------------------
       x            : Random variable in a distribution (unkonwn)
       Distribution : Distribution's function
       Point        : Random variable in the uniform distribution
    --------------------------------------------------------------"""
    return integrate.quad(Distribution,0,x)[0]-Point

def Random_numbers_distribution(f, N, x0 = 0.001, normal = False, args = None):
    """---------------------------------------
    Creates an array of N random numbers with
    a distribution density equal to f. 
    --------------------------------------"""
    if normal == False:
        norm = integrate.quad(f,0,1)[0]
        uf = lambda x: f(x)/norm #Density function with integral=1
    Uniform = random.random(N)
    Map = zeros(N)
    for ii in range(N):
        Map[ii]=fsolve(func,x0,args=(uf,Uniform[ii]))
    return Map

def kepler_galaxy(N, alpha = 0, beta = 0):
    '''--------------------------------------------------------------
    Uses a uniform distrubution of masses to create a plain Disk with 
    a central Black Hole and stars orbiting around it
    -----------------------------------------------------------------
       N            : Number of particles
    --------------------------------------------------------------'''
    # Mass limits [ Solar masses]
    max_mass = 50   # Maximum mass
    min_mass = 1    # Minimum mass 
    BHM = 4e6   # Black Hole mass
    
    random.seed(10) # Seed
    # Generation of N random particles 
    status = empty([N,7])
    # Random masses varies between min_mass and max_mass in solar masses
    status[:-1, 0] = random.random(N-1)*(max_mass-min_mass) + min_mass
    #Random angle generation
    gamma = random.random(N-1)*2*pi
    init_r = 0.4 # Initial radius
    center = [0.5, 0.5, 0.5] # Origin of galaxy
    #Model of density normalized
    f = lambda x: x    
    #Points mapped from the uniform distribution
    Uniform = Random_numbers_distribution(f,N-1)*init_r
    for i in range(N-1):
        #Change to cartesian coordinates
        status[i, 1] = Uniform[i]*(cos(gamma[i])*cos(alpha)+ sin(gamma[i])*cos(beta)*sin(alpha)) + center[0]
        status[i, 2] = Uniform[i]*(sin(gamma[i])*cos(beta)*cos(alpha)-cos(gamma[i])*sin(alpha)) + center[1]
        status[i, 3] = Uniform[i]*sin(gamma[i])*sin(beta) + center[2]
        # Keplerina velocity in the plain of the disc 
        Kep_v = sqrt(G*BHM/Uniform[i])
        vec_vel=array([-(sin(gamma[i])*cos(alpha)-cos(gamma[i])*cos(beta)*sin(alpha)),
                       (cos(gamma[i])*cos(beta)*cos(alpha)+sin(gamma[i])*sin(alpha)), 
                       cos(gamma[i])*sin(beta)])
        status[i, 4] = status[i, 0]*Kep_v*vec_vel[0]
        status[i, 5] = status[i, 0]*Kep_v*vec_vel[1]
        status[i, 6] = status[i, 0]*Kep_v*vec_vel[2]
    # BH's information
    status[N-1, 0] = BHM
    status[N-1, 1:4]=center
    status[N-1, 4:7]=array([0.,0.,0.])
    return status

def bessel_galaxy(N, alpha = 0, beta = 0):
    '''--------------------------------------------------------------
    Use a radial distrubution of masses which is proportional to the 
    brightness surface distributation to create a  Disk resembling an 
    spiral galaxy.
    -----------------------------------------------------------------
       N            : Number of particles
    --------------------------------------------------------------'''
    from scipy.special import kv, iv
    random.seed(10)
    init_r = 0.5  # Initial radius
    # Generates N random particles 
    status = empty([N,7])
    # Random masses varies between min_mass mass and max_mass solar masses
    max_mass = 50
    min_mass = 1
    status[:, 0] = random.random(N)*(max_mass-min_mass) + min_mass
    #Parameters of the model of density of starts (Adimentional)
    Rd = .1
    #Model of density normalized
    f = lambda x: x*exp(-x/Rd)      
    #Random angle generation
    gamma = random.random(N)*2*pi
    #Points mapped from the uniform distribution
    Map = Random_numbers_distribution(f,N, args=(Rd))*init_r
    Rd *= init_r
    center = [0.5, 0.5, 0.5] # Origin of the galaxy  
    for i in range(N):
        #Change to cartesian coordinates
        status[i, 1] = Map[i]*(cos(gamma[i])*cos(alpha)+ sin(gamma[i])*cos(beta)*sin(alpha)) + center[0]
        status[i, 2] = Map[i]*(sin(gamma[i])*cos(beta)*cos(alpha)-cos(gamma[i])*sin(alpha)) + center[1]
        status[i, 3] = Map[i]*sin(gamma[i])*sin(beta) + center[2]
        #Velocity for particles in an exponential disc
        y = Map[i] / (2*Rd)
        sigma = sum(status[:,0])/(2*pi*(Rd**2-(init_r**2+init_r*Rd)*exp(-init_r/Rd)))
        #Magnitud
        Bessel_v = sqrt(4*pi*G*sigma*y**2*(iv(0,y)*kv(0,y)-iv(1,y)*kv(1,y)))
        #Components 
        vec_vel=array([-(sin(gamma[i])*cos(alpha)-cos(gamma[i])*cos(beta)*sin(alpha)),
                       (cos(gamma[i])*cos(beta)*cos(alpha)+sin(gamma[i])*sin(alpha)), 
                       cos(gamma[i])*sin(beta)])
        status[i, 4] = status[i, 0]*Bessel_v*vec_vel[0]
        status[i, 5] = status[i, 0]*Bessel_v*vec_vel[1]
        status[i, 6] = status[i, 0]*Bessel_v*vec_vel[2]
    return status

def spiral_galaxy(N, alpha = 0, beta = 0):
    '''--------------------------------------------------------------
    Use a radial distrubution of masses, which is proportional to the 
    brightness surface distributation to create a plain Bulb and Disk 
    resembling an spiral galaxy
    --------------------------------------------------------------------------
       N            : Number of particles
    ------------------------------------------------------------------------'''
    random.seed(10)
    #Black hole's mass    
    BHM = 4e6  
    # Generates N random particles 
    status = empty([N,7])
    # Random masses varies between min_mass mass and max_mass solar masses
    max_mass = 1
    min_mass = 1
    status[:-1, 0] = random.random(N-1)*(max_mass-min_mass) + min_mass
    #Parameters of the model of density of starts
    const_bulb=2.5
    const_disc=.2
    bulb_radius=0.2
    #Model of density normalized
    f1 = lambda x: exp(-x**(1/4)/const_bulb)        #Bulge
    f2 = lambda x: f1(bulb_radius)*exp(-(x-bulb_radius)/const_disc) #Disc
    f = lambda x:  x*f1(x) if x<bulb_radius else x*f2(x)                #Piecewise
    #Empty array for the points mapped from the uniform distribution
    init_r = 0.4 # Initial radius
    Map = Random_numbers_distribution(f,N,args=(const_bulb, const_disc, bulb_radius))*init_r
    #Random angle generation
    gamma = random.random(N-1)*2*pi
    #Random width
    width = .02
    #Half of with in relation to the radius of the galaxy
    gross  = random.random(N-1)*2*width-width
    temp = beta
    center = [0.5, 0.5, 0.5] # Origin of galaxy   
    for i in range(N-1):
        #Creates an elipsoid in the region of the bulge
        if Map[i] < bulb_radius*init_r:
            a = 0.072
            bulg_countour = a*sqrt(1-(Map[i]/(bulb_radius*init_r))**2)
            gross[i] = random.random(1)*2*bulg_countour-bulg_countour
        #Adjustment for width
        beta += arctan(gross[i]/Map[i])
        Map[i] = sqrt(Map[i]**2+gross[i]**2)
        #Change to cartesian coordinates
        status[i, 1] = Map[i]*(cos(gamma[i])*cos(alpha)+ sin(gamma[i])*cos(beta)*sin(alpha)) + center[0]
        status[i, 2] = Map[i]*(sin(gamma[i])*cos(beta)*cos(alpha)-cos(gamma[i])*sin(alpha)) + center[1]
        status[i, 3] = Map[i]*sin(gamma[i])*sin(beta) + center[2]
        # Keplerina velocity in the plain of the disc
        #Magnitud
        Kep_v = sqrt(G*BHM/Map[i])
        #Components
        vec_vel=array([-(sin(gamma[i])*cos(alpha)-cos(gamma[i])*cos(beta)*sin(alpha)),
                       (cos(gamma[i])*cos(beta)*cos(alpha)+sin(gamma[i])*sin(alpha)), 
                       cos(gamma[i])*sin(beta)])
        status[i, 4] = status[i, 0]*Kep_v*vec_vel[0]
        status[i, 5] = status[i, 0]*Kep_v*vec_vel[1]
        status[i, 6] = status[i, 0]*Kep_v*vec_vel[2]
        beta = temp 
    status[N-1, 0] = BHM
    status[N-1, 1:4] = center
    status[N-1, 4:7] = array([0.,0.,0.])
    return status

def system_init_write(N, model, ini_radius, alpha, beta, format = 'npy', data_folder = 'Data/'):
    '''--------------------------------------------------------------
    Writes a binary file with the initial state of the N-body system. 
    The latter is generated by model. It saves the data in the next
    format:
        [mi, xi, yi, zi, pxi, pyi, pzi] for i=0, ..., N-1
    --------------------------------------------------------------'''
    # Scaling of gravitational constant
    global G
    G *= (0.4/ini_radius)**3

    Numpy_Init = open(data_folder + 'Initial State.' + format, 'wb')
    state = model(N, alpha, beta)
    save(Numpy_Init, state)
    Numpy_Init.close()

def system_init_read(N, format = 'npy', data_folder = 'Data/'):
    '''--------------------------------------------------------------
    Reads a binary file with the initial state of the N-body system.
    And builts the body class.
    --------------------------------------------------------------'''
    bodies = []
    state = load(data_folder + 'Initial State.' + format)
    for i in range(N):
       bodies.append(Node(state[i,0], state[i, 1:4], state[i, 4:7]))
    return bodies

def evolve(bodies, N, n, ini_radius, save_step, data_folder='Data/', format='npy'):
    '''--------------------------------------------------------------
    This function evolves the system in n steps of time using the 
    Verlet algorithm and the Barnes-Hut octo-tree.
    --------------------------------------------------------------'''
    # Scaling of gravitational constant
    global G
    G *= (0.4/ini_radius)**3
    File = open(data_folder + 'Evolution.' + format, 'wb')
    print('Evolution progress:')
    pbar = tqdm(total=n)
    # Principal loop over n time iterations.
    for i in range(n+1):
        # The octo-tree is recomputed at each iteration.
        root = None
        for body in bodies:
            body.reset_location()
            root = add(body, root)

        # Evolution using the Verlet method
        verlet(bodies, root, theta, dt)
        # Save the data in binary files
        if i%save_step==0:
            save_data(File, bodies, N)
            pbar.update(save_step)
    File.close()

def save_data(File, bodies, N):
    '''--------------------------------------------------------------
    Save data of the current state of the bodies into File.
    --------------------------------------------------------------'''
    Data = zeros([N, 7])
    ii = 0
    for body in bodies:
        Data[ii, 0] = body.m
        Data[ii, 1:4] = body.position()
        Data[ii, 4:] = body.momentum
        ii +=1
    save(File, Data)

def read_evolution(N, n, save_step, data_folder='Data/', format='npy', image_folder='imagesBH/'):
    '''--------------------------------------------------------------
    Reads the evolution of the N-body system in n steps of time. 
    Creates images each save_step steps.
    --------------------------------------------------------------'''
    File = open(data_folder + 'Evolution.' + format, 'rb')
    print('Images:')
    pbar = tqdm(total=n//save_step)
    for ii in range(n//save_step):
        state = load(File)
        plot_bodies(state[:, 0], state[:, 1:4], ii, N,image_folder)
        pbar.update(1)

def plot_bodies(m, pos, i, N, image_folder='imagesBH/'):
    '''--------------------------------------------------------------
    Plots images of system's configuration.
    --------------------------------------------------------------'''
    fig = plt.figure(figsize=(10,10), facecolor= 'k')
    ax = fig.gca(projection='3d',facecolor = 'k')
    ax.set_xlim([.08,.92])
    ax.set_ylim([.08,.92])
    ax.set_zlim([.08,.92])

    # Make panes transparent
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane

    # Remove grid lines
    ax.grid(False)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.w_xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

    #ax.view_init(-90, 90)
    Mmax = 50   # Maximum mass
    Mmin = 1    # Minimum mass
    dot_max = 10    # Maximum markersize
    dot_min = 0.1   # Minimun markersize
    for ii in range(N):
        if (m[ii]> Mmax):
            # Black hole
            size = 2*dot_max
            ax.scatter(pos[ii, 0], pos[ii, 1], pos[ii, 2], marker='.', s=size, color='orange')
        else:
            # Stars
            size = ((dot_max-dot_min)*m[ii]+dot_min*Mmax - dot_max*Mmin)/(Mmax-Mmin)
            ax.scatter(pos[ii, 0], pos[ii, 1], pos[ii, 2], marker='.', s=size, color='lightcyan')
    plt.savefig(image_folder+'bodies_{0:06}.png'.format(i))
    plt.close()

def create_video(image_folder='imagesBH/', video_name='my_video.mp4'):
    '''--------------------------------------------------------------
    Creates a .mp4 video using the stored files images
    --------------------------------------------------------------'''
    from os import listdir
    import moviepy.video.io.ImageSequenceClip
    fps = 15
    image_files = [image_folder+img for img in sorted(listdir(image_folder)) if img.endswith('.png')]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)

def tangent_velocity_distribution(N, n, ini_radius, save_step, alpha, beta, data_folder='Data/', format='npy', image_folder='imagesBH/', imag = 1):
    '''--------------------------------------------------------------
    Reads the evolution of the N-body system in n steps of time. 
    Then, writes a binary file with tangent velocity of the bodies, 
    following next format:
        [ri, vti]   for i = 0, ..., N-1.
    --------------------------------------------------------------'''
    # Open evolution file
    File = open(data_folder + 'Evolution.' + format, 'rb')
    # Create tangent velocity file
    Velocity = open(data_folder + 'Tangent_Velocity.' + format, 'wb')
    # Number of times that data was saved
    steps = n//save_step

    # Normal Vector
    normal_vector = array([sqrt(1-tan(alpha)**2)*sin(beta), tan(alpha)*sin(beta), cos(beta)])
    print('Tangent Velocity:')
    pbar = tqdm(steps)
    # Main loop
    factor = ini_radius/0.4
    state = load(File)
    Data = tangent_velocity(state[:, 0], state[:, 1:4], state[:, 4:7], N, normal_vector)
    Data = array(Data)*factor
    Vmax, rmax = max(Data[1]), max(Data[0])
    for ii in range(steps):
        save(Velocity, Data, N)
        if (ii%imag==0):
            # Image's format
            fig = plt.figure(figsize=(10,10))
            plt.title(f'Tangent Velocity at t={"{:.1f}".format((ii*save_step)*0.02)} Gyr')
            ax = fig.gca()
            ax.set_xlabel('Radius, r [kpc]')
            ax.set_ylabel('Tangent Velocity, $V_t$[kpc/Gyr]')
            ax.set_xlim([0, rmax])
            ax.set_ylim([0, Vmax])
            ax.scatter(Data[0], Data[1], marker='+', color='black')
            plt.savefig(image_folder+'TV_{0:06}.png'.format(ii))
            plt.close()
        pbar.update(1)      
        state = load(File)
        Data = tangent_velocity(state[:, 0], state[:, 1:4], state[:, 4:7], N, normal_vector)
        Data = factor*array(Data)
    File.close()
    Velocity.close()

def tangent_velocity(m, pos, momentum, N, normal_vector):
    '''--------------------------------------------------------------
    Computes the tangent speed with respect to orbital plane, which 
    has normal_vector as normal vector. Returns tangent velocity and
    radius. 
    --------------------------------------------------------------'''
    r = zeros(N)
    vt = zeros(N)
    center = [0.5, 0.5, 0.5]
    for i in range(N):
        vel = momentum[i] / m[i] # Velocity
        r[i] = norm(pos[i]-center)   # radius
        vt[i] = dot(vel, cross(pos[i]-center, normal_vector))/ norm(pos[i]-center)
    return r, abs(vt)

def read_model(model):
    '''--------------------------------------------------------------
    Returns the model to create galaxy according to a read input
    --------------------------------------------------------------'''
    if (model=='kepler_galaxy'):
        return kepler_galaxy
    elif (model=='bessel_galaxy'):
        return bessel_galaxy
    elif (model=='spiral_galaxy'):
        return spiral_galaxy
    else:
        raise ValueError(f"Model {model} hasn't been defined.")
