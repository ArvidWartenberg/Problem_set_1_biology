import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

'''
# Plot first derivative of F for some values of r,b to assess stability (just for fun)
r = np.arange(0, 5, 0.1)
b = np.arange(1, 5, 0.1)
R, B = np.meshgrid(r, b)
Z = 1-R*B/(1+R)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(R, B, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
                       
ax.set_zlim(-3, 3)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
print()
'''

# To run the simulation, we define a class for our populations
class Population: # Todo clean up and check documentation
    '''
    @:param N initial population vs. time
    @:param dt time step
    @:param T time delay
    @:param K capacity
    @:param r instantaneous per capita growth
    @:param Allen parameter
    '''
    def __init__(self, K, r, b, N_0):
        self.K = K
        self.r = r
        self.b = b
        self.N_0 = N_0
        #self.N = np.array([N_0])
        #self.delta_N = np.array([0])

    def run_time(self, steps):
        N = np.array([self.N_0])
        for i in range(0, steps):
            N_update = (self.r+1)*(N[-1])/(1+(N[-1]/self.K)**self.b)
            N = np.concatenate((N, np.array([N_update])), axis=0)
        return N

    # Use linear approximation for fixpoint in 0, up to first derivative
    def run_time_linear_approx_N1(self, steps):
        N = np.array([self.N_0])
        N_dot_lin = self.r + 1
        for i in range(0, steps):
            N = np.concatenate((N, np.array([N[-1]*N_dot_lin])), axis=0)
        return N

    def run_time_linear_approx_N2(self, steps):
        N = np.array([self.N_0])
        N_dot_lin = 1-self.r*self.b/(self.r + 1)
        N_fp = self.K*self.r**(1/self.b)
        for i in range(0, steps):
            N = np.concatenate((N, np.array([N_fp + (N[-1] - N_fp)*N_dot_lin])), axis=0)
        return N



# Set parameters for simulations
K = 1e3 # Capacity
r = 1e-1 # Per capita growth rate
b = 1e0 # Punishment parameter for population exceeding K
N_0 = [1, 2, 3, 10]
steps = 130
colors = ['c', 'b', 'r', 'k', 'g', 'm', 'y']


# Study the linear approximation of the system around the fixpoint 0
plt.subplot(1,2,1)
for i in range(len(N_0)):
    population = Population(K=K, r=r, b=b, N_0=N_0[i])
    N = population.run_time(steps=steps)
    N_lin_approx = population.run_time_linear_approx_N1(steps=steps)
    plt.loglog(N, label = '$N_0$ = %i'%(N_0[i]), linewidth=2, c=colors[i])
    plt.loglog(N_lin_approx, linewidth=4, linestyle=':', c=colors[i])#, label='$N_0$ = %i, L-A' % (N_0[i]))
plt.axhline(y=K*r**(1/b), xmin=0, xmax=steps, linestyle=':', label='$N^{*}_{2}=Kr^{1/b}$')
#plt.axis([0, 130, 0, K*r**(1/b)+300])
plt.xlabel('Generation $\\tau$', fontsize=15)
plt.ylabel('Population $N_{\\tau}$', fontsize=15)
plt.title('Model behaviour for different $N_0$', fontsize=20)
plt.legend(prop={"size":12})

# Now we want to analyze the second fixpoint
N_0 = 100 + np.array([-10, -3, -2, -1, 1, 2, 3, 10])
plt.subplot(1,2,2)
for i in range(len(N_0)):
    c = np.random.rand(3, )
    population = Population(K=K, r=r, b=b, N_0=N_0[i])
    N = population.run_time(steps=steps)
    N_lin_approx = population.run_time_linear_approx_N2(steps=steps)
    plt.loglog(N, label = '$N_0$ = %i'%(N_0[i]), linewidth=2, c=c)
    plt.loglog(N_lin_approx, linewidth=4, linestyle=':', c=c)

plt.axhline(y=K*r**(1/b), xmin=0, xmax=steps, linestyle=':', label='$N^{*}_{2}=Kr^{1/b}$')
#plt.axis([0, 130, 0, K*r**(1/b)+300])
plt.xlabel('Generation $\\tau$', fontsize=15)
plt.ylabel('Population $N_{\\tau}$', fontsize=15)
plt.title('Model behaviour for different $N_0$', fontsize=20)
plt.legend(prop={"size":12})
plt.show()


# We observe that as the initial condition (N_0) grows, the approximation is less reliable.
# This is because we will operate further form the point around which we expanded our function
# already from the get-go. The approximation works well while N is not much greater than 0,
# around which we expanded. (this is for the fixpoint at 0)

# Furthermore, we observe somewhat of the same behaviour for the other fixpoint, i.e. the
# behaviour is less and less similar to the true system as the initial condition deviates
# more and more from the point at which we did our linear approximation