import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, alpha, R, eta_0):
        self.alpha = alpha
        self.R = R
        self.eta_0 = eta_0

    def run_time(self, steps):
        eta = np.array([self.eta_0])
        for i in range(0, steps):
            eta_update = self.R*eta[-1]*np.exp(-self.alpha*eta[-1])
            eta = np.concatenate((eta, np.array([eta_update])), axis=0)
        return eta


alpha = 0.01
R_vec = np.arange(1, 30, 0.1)
eta_0 = 900
tau = np.arange(0, 100, 1)

for R in R_vec:
    population = Population(alpha=alpha, R=R, eta_0=eta_0)
    eta = population.run_time(300)[-101:-1]
    plt.scatter(np.ones(len(eta))*R, eta, marker='.', s=.1, c='b', alpha=.6)

plt.axvline(x=5, linestyle=':', c='g', label='Stable equilibrium')
plt.axvline(x=10, linestyle='-.', c='k', label='2 point cycle')
plt.axvline(x=13.5, linestyle='-', c='r', label='4 point cycle')
plt.xlabel('$R$', fontsize=15)
plt.ylabel('$\\eta$', fontsize=15)
plt.title('Last $100$ values of $\\eta$ vs. $R$', fontsize=15)
plt.legend(prop={"size":10}, loc='upper left')
plt.tight_layout(True)


# We see in the plot generated above that R around 5 gives 1 periodicity (constant),
# R~10 gives a 2 periodicity, R~13.5 gives a 4 periodicity. Since the bifurcations
# always doubles the periodicity, there is no 3-periodicity for this model.
# (periodicity <=> cycle)


plt.figure()
steps = 30
population_1 = Population(alpha=alpha, R=5, eta_0=eta_0)
eta_1 = population_1.run_time(steps)
population_2 = Population(alpha=alpha, R=10, eta_0=eta_0)
eta_2 = population_2.run_time(steps)
population_3 = Population(alpha=alpha, R=13.5, eta_0=eta_0)
eta_3 = population_3.run_time(steps)

tau = np.arange(0, len(eta_1))
plt.plot(tau, eta_1, alpha=.3, c='b') # Stable equilibrium
plt.scatter(tau, eta_1, alpha=1, c='b', marker='o', label='Stable quilibrium, $R=5$') # Stable equilibrium
plt.plot(tau, eta_2, alpha=.3, c='r') # 2 point cycle
plt.scatter(tau, eta_2, alpha=1, c='r', marker='^', label='2 point cycle, $R=10$') # 2 point cycle
plt.plot(tau, eta_3, alpha=.3, c='k') # 4 point cycle
plt.scatter(tau, eta_3, alpha=1, c='k', marker='x', label='4 point cycle, $R=13.5$') # 4 point cycle
plt.xlabel('$\\tau$', fontsize=15)
plt.ylabel('$N_{\\tau}$', fontsize=15)
plt.title('System behaviour for different $R$', fontsize=15)
plt.legend(prop={"size":12})
plt.tight_layout(True)
plt.legend()
plt.show()


