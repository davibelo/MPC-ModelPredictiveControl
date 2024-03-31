import numpy as np
import control as ct
from src.mpc import calculate_step_responses, simulateMIMO
from src.mpc import plot_step_responses, plot_simulation_results
from src.mpc import mpc_controller_scipy_minimize
from testSystems import *
import joblib

# Initial parameters
nsimTest = 1000  # Test simulation time in sampling periods
nsim = 500  # Simulation time in sampling periods
T = 1  # Sampling time (min)
ny = 2
nu = 2
u0 = np.array([166., 10648.], dtype=float)
y0 = np.array([0.0139689, 0.0152698], dtype=float)

#load the step responses
Gmstep = joblib.load('outputs/Gstep.joblib')
Gpstep = Gmstep
plot_step_responses(Gmstep, nsim, T, plot_max_length=120, fig_name='LPGdebutanizer - Step Responses')

# Set Simulation test
tsimTest = np.linspace(0, nsimTest * T, nsimTest + 1)  # Simulation Time vector
Utest = np.tile(u0, (len(tsimTest), 1)).T.astype(float)  # Initialisation of the input matrix
Utest[:, 200:] = np.array([167., 10648.]).reshape(-1, 1).astype(float)  # Step change in inputs
Utest[:, 400:] = np.array([166., 10648.]).reshape(-1, 1).astype(float)  # Step change in inputs
Utest[:, 600:] = np.array([166., 10748.]).reshape(-1, 1).astype(float)  # Step change in inputs
Utest[:, 800:] = np.array([166., 10648.]).reshape(-1, 1).astype(float)  # Step change in inputs

# Simulate the test
YsimTest, delta_U = simulateMIMO(Gmstep, tsimTest, ny, nu, y0, u0, Utest)

# Call the function to plot simulation results
plot_simulation_results(YsimTest, Utest, tsimTest, 'LPGdebutanizer - test')

# MPC parameters
n = 120  # Stabilizing horizon
p = 120  # Output prediction horizon
m = 5  # Control horizon
umax = np.array([200., 20000.])
umin = np.array([120., 1000.])
ymax = np.array([0.02, 0.01])
ymin = np.array([0.01, 0.005])
dumax = np.array([1, 1000.])
q = np.array([1000., 1000.])  # Output weights
r = np.array([100., 0.00001])  # Input weights

# Control loop simulation
tsim = np.linspace(0, nsim * T, nsim + 1)  # Simulation Time vector
Ysim = np.tile(y0, (len(tsim), 1)).T.astype(float) # Initialize Y_sim
Usim = np.tile(u0, (len(tsim), 1)).T.astype(float)  # Initialize U_sim
for t in range(nsim + 1):
    if t == 0:
        u0temp = u0
        y0temp = y0
    else:
        u_opt = mpc_controller_scipy_minimize(ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q,
                                              r, u0temp, y0temp, Gmstep)
        # Update Usim from t to the end with u_opt
        Usim[:, t:] = u_opt.reshape(-1, 1)
        # Simulate
        Ysim, delta_U = simulateMIMO(Gpstep, tsim, ny, nu, y0, u0, Usim)

        # Update Y0 and U0 for the next step
        y0temp = Ysim[:, t]
        u0temp = Usim[:, t]
    print('y: ', y0temp)
    print('u: ', u0temp)
    print('delta_U: ', delta_U[:, t])
    print(f'Step {t} of {nsim} completed')
plot_simulation_results(Ysim, Usim, tsim, 'LPGdebutanizer - control loop', ymin, ymax, umin, umax)
