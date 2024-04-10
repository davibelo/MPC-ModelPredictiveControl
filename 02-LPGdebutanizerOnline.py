import time
import numpy as np
import joblib
import subprocess
from src.mpc import mpc_controller_scipy_minimize

# Time parameters
REAL_TIME_SYNC_FATOR = 0.1 # Real time sync factor on simulator
cycle_time = 60*REAL_TIME_SYNC_FATOR  # Cycle time in seconds

# MPC parameters
T = 1  # Sampling time (min)
ny = 2 # number of outputs
nu = 2 # number of inputs
n = 120  # Stabilizing horizon
p = 120  # Output prediction horizon
m = 5  # Control horizon
umax = np.array([200., 20000.])
umin = np.array([120., 1000.])
ymax = np.array([0.015, 0.016]) # run initials
ymin = np.array([0.013, 0.015]) # run initials
# ymax = np.array([0.015, 0.010]) # run 1
# ymin = np.array([0.013, 0.009]) # run 1
ymax = np.array([0.013, 1]) # run 4
ymin = np.array([0.011, 0]) # run 4
dumax = np.array([1, 1000.])
q = np.array([1000., 1000.])  # Output weights
r = np.array([100., 0.001])  # Input weights

# OPC parameters
OPC_SERVER = 'Matrikon.OPC.Simulation.1'
HOST = 'localhost'

#load the step responses
Gmstep = joblib.load('outputs/Gstep.joblib')

# Initialize SPs with PV values
tags_to_read = ['APD.TBOTTOMPV', 'APD.QREFLUXPV']

# Reading plant with OpenOPC CLI
command = 'C:\\Program Files\\OpenOPC\\bin\\opc.exe'
server_args = [f'--server={OPC_SERVER}', f'--host={HOST}']
tags_to_read_args = ['--read'] + tags_to_read + ['--output=values']
cmd = [command] + server_args + tags_to_read_args
result = subprocess.run(cmd, capture_output=True, text=True)
output = result.stdout
lines = output.split('\n')
TBOTTOMPV, QREFLUXPV = float(lines[0]), float(lines[1])
print('Initializing SPs with PV values...')
print(f'TBOTTOMPV: {TBOTTOMPV}')
print(f'QREFLUXPV: {QREFLUXPV}\n')

TBOTTOMSP = TBOTTOMPV
QREFLUXSP = QREFLUXPV
TBOTTOMSP = 166 # initial SP on saved snapshot
QREFLUXSP = 10648 # initial SP on saved snapshot

# Writing to plant with OpenOPC CLI
write_args = ['--write', 'APD.TBOTTOMSP', str(TBOTTOMSP), 'APD.QREFLUXSP', str(QREFLUXSP)]
cmd = [command] + write_args
result = subprocess.run(cmd, capture_output=True, text=True)
print('writing tags initial SPs...')
print('TBOTTOMSP:', TBOTTOMSP)
print('QREFLUXSP:', QREFLUXSP)

# Wait for user input to start
print('SPs initialized!')
input('Press Enter to continue...')

while True:
    start_time = time.time()  # Get the current time
    # Reading plant with OpenOPC CLI
    command = 'C:\\Program Files\\OpenOPC\\bin\\opc.exe'
    server_args = [f'--server={OPC_SERVER}', f'--host={HOST}']
    tags_to_read = ['APD.C2LPG', 'APD.C5LPG', 'APD.TBOTTOMSP', 'APD.QREFLUXSP']
    tags_to_read_args = ['--read'] + tags_to_read + ['--output=values']
    cmd = [command] + server_args + tags_to_read_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    lines = output.split('\n')
    C2LPG, C5LPG, TBOTTOMSP, QREFLUXSP = float(lines[0]), float(lines[1]), float(lines[2]), float(
        lines[3])
    print('reading tags...')
    print(f'C2LPG: {C2LPG}')
    print(f'C5LPG: {C5LPG}')
    print(f'TBOTTOMSP: {TBOTTOMSP}')
    print(f'QREFLUXSP: {QREFLUXSP}\n')

    # Call MPC
    y0temp = np.array([C2LPG, C5LPG], dtype=float)
    u0temp = np.array([TBOTTOMSP, QREFLUXSP], dtype=float)
    u_opt = mpc_controller_scipy_minimize(ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r,
                                          u0temp, y0temp, Gmstep)
    u_opt = u_opt.reshape(-1, 1)
    TBOTTOMSP = u_opt[0].item()
    QREFLUXSP = u_opt[1].item()
    print('writing tags...')
    print('TBOTTOMSP:', TBOTTOMSP)
    print('QREFLUXSP:', QREFLUXSP)

    # Writing to plant with OpenOPC CLI
    write_args = ['--write', 'APD.TBOTTOMSP', str(TBOTTOMSP), 'APD.QREFLUXSP', str(QREFLUXSP)]
    cmd = [command] + write_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    end_time = time.time()

    # Calculate sleep time
    elapsed_time = end_time - start_time
    time_to_wait = cycle_time - elapsed_time
    print(f'Elapsed time: {elapsed_time} seconds')
    print(f'Waiting {time_to_wait} seconds...\n')
    time.sleep(time_to_wait)
