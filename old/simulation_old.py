'''
Example 1 :
Debutanizer system
'''

import numpy as np
import control as ct
from mpc import *
import matplotlib.pyplot as plt


# Parameters
nu = 3  # Number of manipulated inputs
ny = 2  # Number of controlled outputs
T = 1  # Sampling time (min)
n = 300  # Stabilizing horizon
p = 30  # Output prediction horizon
m = 3  # Control horizon
nsim = 2  # Simulation time in sampling periods
nsimTest = 2000  # Simulation time for testing in sampling periods

# Input and output constraints
umax = np.array([950, 9.0, 1e3])
umin = np.array([400, 3.0, -1e3])
ymax = np.array([2.8, 0.75])
ymin = np.array([2.0, 0.65])
dumax = np.array([10, 0.5, 1e-4])

# Weights of the control layer
q = np.array([10000, 10000]) # Output weights
r = np.array([100, 100, 1])  # Input weights

# Weights of the economic layer
py = np.array([0, 0])       # Output weights
pu = np.array([0, 1, 0])    # Input weights
peps = np.array([1e5, 1e5]) # Penalty weights
ru = np.array([1, 1, 0])    # Input optimization weights

# Initial input and output values
u0 = np.array([700, 6.2, 0])
y0 = np.array([2.5, 0.7])
# y0 = np.array([3, 0.5]) # y0 out of control limits

# System Transfer functions
G11 = ct.tf([-1.9973e-3, -1.3105e-4], [1, -8.3071e-1, -5.4544e-1, 5.1700e-1, 0], dt=T)
G12 = ct.tf([1.9486e-2 / 24, 4.6325e-2 / 24],
            [1, -1.5119, 4.3596e-1, 8.2888e-2, 0, 0, 0, 0, 0, 0, 0],
            dt=T)
G13 = ct.tf([-1.5789e-4, -8.3160e-5],
            [1, -6.9794e-1, -1.8196e-1, -9.3550e-2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dt=T)

G21 = ct.tf([9.2722e-5, 3.1602e-5], [1, -1.1613, -7.5733e-2, 3.7749e-1, 0, 0, 0, 0], dt=T)

G22 = ct.tf([-4.2379e-2 / 24], [1, -1.3953, 2.5570e-1, 1.5459e-1, 0, 0, 0, 0, 0, 0, 0], dt=T)

G23 = ct.tf([6.9654e-6, 8.6757e-6], [1, -8.8584e-1, 0, 0, 0, 0, 0, 0, 0, 0], dt=T)

# Create a 2D list of transfer functions
Gm = [[G11, G12, G13],
      [G21, G22, G23]]

#TODO: when simulating closed loop, use a slightly different system to simulate modelling imperfections
Gp = Gm  # System = Model

Gmstep = calculate_step_responses(Gm, max(nsim, nsimTest), T)

# Simulation test
tsimTest = np.linspace(0, nsimTest * T, nsimTest + 1)  # Simulation Time vector
Utest = np.tile(u0, (len(tsimTest), 1)).T # Initialisation of the input matrix
Utest[:, 100:] = np.array([710, 6.2, 0]).reshape(-1, 1)  # Step change in inputs
Utest[:, 500:] = np.array([710, 6.3, 0]).reshape(-1, 1)  # Step change in inputs
Utest[:, 1000:] = np.array([710, 6.3, 10]).reshape(-1, 1)  # Step change in inputs
Utest[:, 1500:] = np.array([700, 6.2, 0]).reshape(-1, 1)  # Step change in inputs

YsimTest = simulateMIMO(Gmstep, tsimTest, ny, nu, y0, Utest)

# Call the function to plot simulation results
plot_simulation_results(YsimTest, Utest, tsimTest, 'simulation_test')

# Control loop
# Gpstep = calculate_step_responses(Gp, nsim, T)
# tsim = np.linspace(0, nsim * T, nsim + 1)  # Simulation Time vector
# Y_sim = np.zeros((ny, len(tsim))) # Initialize Y_sim
# U_sim = np.zeros((nu, len(tsim))) # Initialize U_sim
# for t in range(nsim):
#     if t == 0:
#         U0 = u0
#         Y0 = y0
#     U = mpc_controller(ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r, U0, Y0, Gmstep)
#     # Simulate one step
#     Y = simulateMIMO(Gpstep, np.array([t * T, (t + 1) * T]), ny, nu, Y0, U)
#     # Append Y to Y_sim
#     Y_sim[:, t] = Y[-1]
#     # Append U to U_sim
#     U_sim[:, t] = U[-1]
#     # Update Y0 and U0 for the next step
#     Y0 = Y[-1]
#     U0 = U[-1]
