import time
import subprocess
from src.mpc import mpc_controller_scipy_minimize

# MPC parameters
EXECUTION_TIME = 30 # Execution time in seconds
T = 1  # Sampling time (min)
ny = 2
nu = 2
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

# OPC parameters
OPC_SERVER = 'Matrikon.OPC.Simulation.1'
HOST = 'localhost'
tags_to_read = ['APD.C2LPG', 'APD.C5LPG', 'APD.TBOTTOMSP', 'APD.QREFLUXSP']

while True:
    start_time = time.time()  # Get the current time
    # Reading plant with OpenOPC CLI
    command = 'C:\\Program Files\\OpenOPC\\bin\\opc.exe'
    server_args = [f'--server={OPC_SERVER}', f'--host={HOST}']
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
    TBOTTOMSP, QREFLUXSP = u_opt.reshape(-1, 1)
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
    time_to_wait = EXECUTION_TIME - elapsed_time
    print(f'Elapsed time: {elapsed_time} seconds')
    print(f'Waiting {time_to_wait} seconds...\n')
    time.sleep(time_to_wait)
