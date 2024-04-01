import subprocess

OPC_SERVER = 'Matrikon.OPC.Simulation.1'
HOST = 'localhost'
tags_to_read = ['APD.C2LPG', 'APD.C5LPG', 'APD.TBOTTOMSP', 'APD.QREFLUXSP']

# Define the command and arguments
command = 'C:\\Program Files\\OpenOPC\\bin\\opc.exe'
server_args = [f'--server={OPC_SERVER}', f'--host={HOST}']
tags_to_read_args = ['--read'] + tags_to_read + ['--output=values']

# Combine the command and arguments into a single list
cmd = [command] + server_args + tags_to_read_args

# Execute the command and capture the output
result = subprocess.run(cmd, capture_output=True, text=True)

# The stdout attribute contains the output
output = result.stdout
lines = output.split('\n')

# Assign each line to a separate variable
C2LPG, C5LPG, TBOTTOMSP, QREFLUXSP = float(lines[0]), float(lines[1]), float(lines[2]), float(lines[3])

# Print the output
print(f'C2LPG: {C2LPG}')
print(f'C5LPG: {C5LPG}')
print(f'TBOTTOMSP: {TBOTTOMSP}')
print(f'QREFLUXSP: {QREFLUXSP}')

# New values to write
TBOTTOMSP = 120.
QREFLUXSP = 670.

tags_to_write = ['APD.TBOTTOMSP', 'APD.QREFLUXSP']
variables_to_write = [TBOTTOMSP, QREFLUXSP]

# Arguments to write new values
write_args = ['--write', 'APD.TBOTTOMSP', float(TBOTTOMSP), 'APD.QREFLUXSP', float(QREFLUXSP)]

# Combine the command and arguments into a single list
cmd = [command] + write_args

# Execute the command
result = subprocess.run(cmd, capture_output=True, text=True)

# Print the output
print(result.stdout)
