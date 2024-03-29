import numpy as np
import control as ct
from src.mpc import calculate_step_responses, simulateMIMO, plot_simulation_results, mpc_controller_scipy_minimize
from testSystems import *

nsim = 500  # Simulation time in sampling periods

Gmstep = joblib.load('outputs/Gmstep.joblib')
Gpstep = Gmstep
