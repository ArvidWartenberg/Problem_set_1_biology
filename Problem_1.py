import numpy as np
import matplotlib.pyplot as plt


class Population:
    '''
    @:param N initial population vs. time
    @:param dt time step
    @:param T time delay
    @:param K capacity
    @:param r instantaneous per capita growth
    @:param Allen parameter
    '''
    def __init__(self, N_0, dt, T, K, r, A):
        self.T_step = int(T/dt)
        self.N = np.ones(self.T_step+1) * N_0
        self.N_dot = np.zeros(self.T_step+1)
        self.dt = dt
        self.T = T
        self.K = K
        self.r = r
        self.A = A

    def run_time(self, steps):
        for i in range(0, steps):
            N_dot = r * self.N[-1] * (1 - self.N[-1 - self.T_step] / self.K) * (self.N[-1] / self.A - 1)
            N_new = self.N[-1]+N_dot*self.dt
            self.N_dot = np.concatenate((self.N_dot, np.array([N_dot])), axis=0)
            self.N = np.concatenate((self.N, np.array([N_new])), axis=0)



    def get_time(self):
        return np.arange(0, len(self.N), 1)*self.dt


# Set parameters
dt = 0.01
A = 20
K = 100
r = 0.1
N_0 = 50
simulation_steps = 10000



# To show some different characteristics of the model,
# we study the following choices of T, which will illustrate
# no oscillations (T=0.1), damped oscillations (T=2) and
# stable oscillations (T=2)
T = [0.1, 1.5, 4]
for T_i in T:

    plt.subplot(1, 2, 1)
    if T_i == 4:
        plt.subplot(1, 2, 2)
        simulation_steps = 40000

    population = Population(N_0, dt, T_i, K, r, A)
    population.run_time(simulation_steps)

    t = population.get_time()
    N = population.N
    N_dot = population.N_dot

    plt.plot(t, N, label='T=%.2f'%(T_i), linewidth=2, alpha=1)
    plt.title('Model behaviour for different time delays T', fontsize=15)
    plt.xlabel('Time [unit time]', fontsize=15)
    plt.ylabel('Population size [members]', fontsize=15)
    #plt.subplot(1, 2, 2)
    #plt.plot(t, N_dot, label='T=%.2f'%(T_i), linewidth=2, alpha=1)
    #plt.title('Model behaviour for different time delays T', fontsize=15)
    #plt.xlabel('Time [unit time]', fontsize=12)
    #plt.ylabel('Population growth [members/unit time]', fontsize=12)
plt.subplot(1, 2, 1)
plt.legend(prop={"size":10}, loc='lower right')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.legend(prop={"size":10}, loc='lower right')
plt.grid(True)


# We now want to assess at which value of T the model starts
# exhibiting damped behaviour... To do this, we increment T
# in a while loop, and stop when the derivative N dot goes below
# zero at some point. This corresponds to a dampening motion in N.

# We will increment T with the following:
T_increment = 0.01
T_damp = 0.1

# We also define a tolerance, i.e. the derivative must
# be sufficiently below zero (to avoid numerical problems)
tol = 0.01

# Using results from a), we deem it sufficient to run
# 2000 iterations..

while True:
    population = Population(N_0, dt, T_damp, K, r, A)
    population.run_time(2000)

    t = population.get_time()
    N_dot = population.N_dot


    if min(N_dot)< -tol:
        break

    # Increment T
    T_damp += T_increment

print('The model starts exhibiting dampening behaviour for T > %.3f'%(T_damp))


# We now want to find the value T_H such that the fixpoint changes its behaviour
# from attracting (dampening oscillation) to a limit cycle (stable oscillation)

# We analytically determined T_H to 3.927. So lets look for a limit cycle
# for choices of T close to this...

T_H = [3.9, 4.0, 3.9, 4.0]
simulation_steps = [50000, 50000, 150000, 150000]
plt.figure()
for i in [1, 2, 3, 4]:
    plt.subplot(2,2,i)
    population = Population(N_0, dt, T_H[i-1], K, r, A)
    population.run_time(simulation_steps[i-1])

    t = population.get_time()
    N = population.N
    N_dot = population.N_dot

    plt.plot(N, N_dot, label='T=%.2f'%(T_H[i-1]), linewidth=2, alpha=1)
    plt.title('%i e4 simulated steps with T=%.1f'%(int(simulation_steps[i-1]/10000), T_H[i-1]), fontsize=12)
    plt.xlabel('$N$ [members]', fontsize=12)
    plt.ylabel('$\dfrac{dN}{dt}$ [members/unit time]', fontsize=12)
    plt.legend()
    plt.grid(True)
plt.tight_layout(True)
plt.show()




plt.show()


print()
