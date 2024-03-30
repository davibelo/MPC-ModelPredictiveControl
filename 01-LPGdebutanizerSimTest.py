import numpy as np
import control as ct
from src.mpc import calculate_step_responses, simulateMIMO
from src.mpc import plot_step_responses, plot_simulation_results
from src.mpc import mpc_controller_scipy_minimize
from testSystems import *
import joblib

nsimTest = 4000  # Test simulation time in sampling periods
nsim = 500  # Simulation time in sampling periods
T = 1  # Sampling time (min)

ny = 2
nu = 2
u0 = np.array([166, 10648])
y0 = np.array([0.0139689, 0.0152698])

Gmstep = joblib.load('outputs/Gstep.joblib')
Gpstep = Gmstep

plot_step_responses(Gmstep, nsim, T, plot_max_length=120, fig_name='LPGdebutanizer - Step Responses')

tsimTest = np.linspace(0, nsimTest * T, nsimTest + 1)  # Simulation Time vector
Utest = np.tile(u0, (len(tsimTest), 1)).T  # Initialisation of the input matrix
Utest[:, 250:] = np.array([167, 10648]).reshape(-1, 1)  # Step change in inputs
Utest[:, 1000:] = np.array([167, 10649]).reshape(-1, 1)  # Step change in inputs
Utest[:, 2000:] = np.array([166, 10649]).reshape(-1, 1)  # Step change in inputs
Utest[:, 3000:] = np.array([166, 10648]).reshape(-1, 1)  # Step change in inputs

# Simulate the test
YsimTest, delta_U = simulateMIMO(Gmstep, tsimTest, ny, nu, y0, u0, Utest)

# Call the function to plot simulation results
plot_simulation_results(YsimTest, Utest, tsimTest, 'LPGdebutanizer - test')
