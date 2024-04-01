import subprocess

OPC_SERVER = 'Matrikon.OPC.Simulation.1'
HOST = 'localhost'
tags_to_read = ['APD.C2LPG', 'APD.C5LPG', 'APD.TBOTTOMSP', 'APD.QREFLUXSP']

# Define the command and arguments
command = 'C:\\Program Files\\OpenOPC\\bin\\opc.exe'
args = [f'--server={OPC_SERVER}', f'--host={HOST}', '--read'] + tags_to_read

# Combine the command and arguments into a single list
cmd = [command] + args

# Execute the command and capture the output
result = subprocess.run(cmd, capture_output=True, text=True)

# The stdout attribute contains the output
output = result.stdout

# Print the output
print(output)
