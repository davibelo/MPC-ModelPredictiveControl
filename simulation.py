import numpy as np
import control as ct
from src.mpc import calculate_step_responses, simulateMIMO, plot_simulation_results, mpc_controller_scipy_minimize
from systems import *


nsim = 500  # Simulation time in sampling periods
nsimTest = 2000  # Test simulation time in sampling periods

# Instantiate the system
s = System1()

# Calculate step responses
Gmstep = calculate_step_responses(s.Gm, nsimTest, s.T, 'step responses - test')
Gpstep = calculate_step_responses(s.Gm, nsim, s.T, 'step responses - control loop simulation') # Gp = Gm for this example

# Set Simulation test
Utest, tsimTest = s.generate_test_input(nsimTest)  # Generate test input

# Simulate the test
YsimTest, delta_U = simulateMIMO(Gmstep, tsimTest, s.ny, s.nu, s.y0, s.u0, Utest)

# Call the function to plot simulation results
plot_simulation_results(YsimTest, Utest, tsimTest, 'plant - test')

# Control loop simulation
tsim = np.linspace(0, nsim * s.T, nsim + 1)  # Simulation Time vector
Ysim = np.tile(s.y0, (len(tsim), 1)).T # Initialize Y_sim
Usim = np.tile(s.u0, (len(tsim), 1)).T  # Initialize U_sim
for t in range(nsim + 1):
    if t == 0:
        u0temp = s.u0
        y0temp = s.y0
    else:
        u_opt = mpc_controller_scipy_minimize(s.ny, s.nu, s.T, s.n, s.p, s.m, s.umax, s.umin, s.ymax, s.ymin, s.dumax, s.q,
                                              s.r, u0temp, y0temp, Gmstep)
        # Update Usim with u_opt
        for i in range(Usim.shape[0]):
            Usim[i, t:] = u_opt[i]
        # Simulate
        Ysim, delta_U = simulateMIMO(Gpstep, tsim, s.ny, s.nu, s.y0, s.u0, Usim)
        # Update Y0 and U0 for the next step
        y0temp = Ysim[:, t]
        u0temp = Usim[:, t]
    print(f'Step {t} of {nsim} completed')
plot_simulation_results(Ysim, Usim, tsim, 'plant - control loop', s.ymin, s.ymax, s.umin, s.umax)
