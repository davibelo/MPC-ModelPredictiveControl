import numpy as np
import control as ct
from src.mpc import calculate_step_responses, simulateMIMO, plot_simulation_results, mpc_controller_scipy_minimize
from testSystems import *


nsim = 500  # Simulation time in sampling periods
nsimTest = 2000  # Test simulation time in sampling periods

# Instantiate the system
s1 = System1()

# Calculate step responses
Gmstep = calculate_step_responses(s1.Gm, nsimTest, s1.T, 'step responses - test')
Gpstep = calculate_step_responses(s1.Gm, nsim, s1.T, 'step responses - control loop simulation') # Gp = Gm for this example

# Set Simulation test
Utest, tsimTest = s1.generate_test_input(nsimTest)  # Generate test input

# Simulate the test
YsimTest, delta_U = simulateMIMO(Gmstep, tsimTest, s1.ny, s1.nu, s1.y0, s1.u0, Utest)

# Call the function to plot simulation results
plot_simulation_results(YsimTest, Utest, tsimTest, 'plant - test')

# Control loop simulation
tsim = np.linspace(0, nsim * s1.T, nsim + 1)  # Simulation Time vector
Ysim = np.tile(s1.y0, (len(tsim), 1)).T.astype(float) # Initialize Y_sim
Usim = np.tile(s1.u0, (len(tsim), 1)).T.astype(float)  # Initialize U_sim
for t in range(nsim + 1):
    if t == 0:
        u0temp = s1.u0
        y0temp = s1.y0
    else:
        u_opt = mpc_controller_scipy_minimize(s1.ny, s1.nu, s1.T, s1.n, s1.p, s1.m, s1.umax,
                                              s1.umin, s1.ymax, s1.ymin, s1.dumax, s1.q, s1.r,
                                              u0temp, y0temp, Gmstep)
        # Update Usim from t to the end with u_opt
        Usim[:, t:] = u_opt.reshape(-1, 1)
        # Simulate
        Ysim, delta_U = simulateMIMO(Gpstep, tsim, s1.ny, s1.nu, s1.y0, s1.u0, Usim)
        # Update Y0 and U0 for the next step
        y0temp = Ysim[:, t]
        u0temp = Usim[:, t]
    print('y: ', y0temp)
    print('u: ', u0temp)
    print('delta_U: ', delta_U[:, t])
    print(f'Step {t} of {nsim} completed')
plot_simulation_results(Ysim, Usim, tsim, 'plant - control loop', s1.ymin, s1.ymax, s1.umin, s1.umax)
