import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time


def calculate_step_responses(Gm, nsim, T, fig_name, plot_max_lenght=500):
    """
    Calculate and plot the step responses of a set of systems.

    Parameters:
    Gm (list of list of control.TransferFunction): A 2D list of systems represented as transfer functions.
    nsim (int): The number of simulation steps.
    T (float): The time interval between simulation steps.

    Returns:
    Gstep (list of list of numpy.ndarray): A 2D list of step responses. Each step response is a 1D numpy array of the same length as the simulation time.

    This function calculates the step responses of the systems in Gm over the simulation time defined by nsim and T. It then plots the step responses and saves the plot as 'step_responses.png'. The calculated step responses are returned as a 2D list of 1D numpy arrays.
    """
    # Calculate step responses
    tstep = np.linspace(0, nsim * T, nsim + 1)
    Gstep = [[None for _ in row] for row in Gm]

    for j, row in enumerate(Gm):
        for i, G in enumerate(row):
            t_out, y_out = ct.step_response(G, tstep)
            Gstep[j][i] = y_out

    # Plot step responses
    plot_step_responses(Gstep, nsim, T, fig_name)

    return Gstep

def plot_step_responses(Gstep, nsim, T, fig_name):
    tstep = np.linspace(0, nsim * T, nsim + 1)
    nrows = len(Gstep)
    ncols = len(Gstep[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
    fig.suptitle('step_responses')
    for j, row in enumerate(Gstep):
        for i, y_out in enumerate(row):
            axs[j, i].plot(tstep, y_out)
            axs[j, i].set_title(f'Gstep[{j+1}][{i+1}]')
            axs[j, i].grid(True)
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Step Response')
    plt.savefig(f'figures/{fig_name}.png')

def plot_simulation_results(Ysim, U, tsim, fig_name, ymin=None, ymax=None, umin=None, umax=None):
    # Plot simulation results
    nrows = max(len(U), len(Ysim))
    fig, axs = plt.subplots(nrows, 2, figsize=(15, 15))
    fig.suptitle(fig_name)
    for i, y_out in enumerate(Ysim):
        axs[i, 0].plot(tsim, y_out)
        axs[i, 0].set_title(f'Ysim[{i}]')
        axs[i, 0].grid(True)
        if ymin is not None and ymax is not None:
            axs[i, 0].hlines([ymin[i], ymax[i]], tsim[0], tsim[-1], colors='r', linestyles='dotted')
    for i, u_out in enumerate(U):
        axs[i, 1].plot(tsim, u_out)
        axs[i, 1].set_title(f'U[{i}]')
        axs[i, 1].grid(True)
        if umin is not None and umax is not None:
            axs[i, 1].hlines([umin[i], umax[i]], tsim[0], tsim[-1], colors='r', linestyles='dotted')
    for ax in axs[-1]:
        ax.set(xlabel='Time')
    for ax in axs[:, 0]:
        ax.set(ylabel='Ysim')
    for ax in axs[:, 1]:
        ax.set(ylabel='U')
    plt.savefig(f'figures/{fig_name}.png')


def simulateMIMO(Gstep, tsim, ny, nu, y0, u0, U):
    """
    Simulate the MIMO system response y given the step response matrix Gstep, time vector tsim,
    number of outputs ny, number of inputs nu, initial condition y0, initial condition u0, and input matrix U, 
    with the first element of U considered as u0 for each input and delta_U calculated accordingly.

    Parameters:
    - Gstep: The 3D array (or list that can be converted to an array) of the system's step response
             for each output to each input (dimensions [ny, nu, t]).
    - tsim: Time vector for the simulation (array).
    - ny: Number of outputs.
    - nu: Number of inputs.
    - y0: Initial condition of y (array with dimensions [ny]).
    - u0: Initial condition of u (array with dimensions [nu]).
    - U: Matrix of input vectors at each time step (array with dimensions [nu, len(tsim)]) to simulate
    
    Returns:
    - Y: The simulated response of the system (array with dimensions [ny, len(tsim)]).
    - delta_U: The difference between the input at each time step and the previous time step (array with dimensions [nu, len(tsim)]).
    """
    Gstep = np.array(Gstep)
    # Check if Gstep is not the same length as tsim
    tsim_len = len(tsim)
    if Gstep.shape[2] != tsim_len:    
        pad_width = tsim_len - Gstep.shape[2] # Calculate the number of padding elements needed
        pad_widths = ((0, 0), (0, 0), (0, pad_width)) # Create a tuple defining the padding for each dimension
        Gstep = np.pad(Gstep, pad_widths, mode='edge') # Extend Gstep to match the length of tsim
    Y = np.zeros((ny, tsim_len), dtype=float)
    delta_U = np.zeros_like(U).astype(float)
    U = np.array(U, dtype=float)  # Ensure U is a numpy float array
    U = np.hstack((u0[:, np.newaxis], U))  # Add u0 at the beginning of U
    for t in range(tsim_len):
        for j in range(ny):
            y = y0[j]
            for i in range(nu):
                # Calculate delta_U as the difference from the previous time step
                delta_U[i, t] = U[i, t + 1] - U[i, t]
                # Perform convolution only if t > 0
                if t > 0:
                    y += np.sum(
                        np.convolve(Gstep[j, i, :t + 1],
                                    np.full(t + 1, delta_U[i, :t + 1]),
                                    mode='valid'))
                else:
                    y += delta_U[i, t] * Gstep[j, i, 0]
            Y[j, t] = y
    return Y, delta_U


def mpc_controller_scipy_minimize(ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r, u0temp,
                                  y0temp, Gstep):
    # Define the control objective function
    def control_objective(U_flatted, *args):
        ''' Control objective function '''
        ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r, u0temp, y0temp, Gstep = args

        # Reshape U back to its original shape
        U = U_flatted.reshape(nu, m)

        # Resize U to p dimension padding with the last value
        pad_width = p - m
        if pad_width > 0:
            U = np.pad(U, ((0, 0), (0, pad_width)), mode='edge')

        # Simulate the system
        tp = np.linspace(0, p * T, p)
        Yp, delta_U = simulateMIMO(Gstep, tp, ny, nu, y0temp, u0temp, U)

        # Cost - Penalize the deviation from the setpoint
        errors_above_ymax = np.zeros_like(Yp)
        errors_below_ymin = np.zeros_like(Yp)
        for i in range(Yp.shape[0]):  # For each variable
            for j in range(Yp.shape[1]):  # For each step
                if Yp[i, j] > ymax[i]:
                    errors_above_ymax[i, j] = Yp[i, j] - ymax[i]
                elif Yp[i, j] < ymin[i]:
                    errors_below_ymin[i, j] = ymin[i] - Yp[i, j]
        errors_above_ymax *= q[:, np.newaxis]
        errors_below_ymin *= q[:, np.newaxis]
        J = np.sum(errors_above_ymax) + np.sum(errors_below_ymin)

        # Cost - Penalize the control moves
        delta_U = np.absolute(delta_U)
        r = r[:3, np.newaxis]
        J += np.sum(delta_U * r)

        return J

    start_time = time.time()  # Start timing

    # Prepare U to be optimized
    U = np.tile(u0temp, (m, 1)).T.astype(float)  # initialize with the u0temp
    U_flatted = U.flatten()  # flatten U

    # Define the arguments for the control objective function
    args = (ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r, u0temp, y0temp, Gstep)

    # Define the bounds for U
    bounds = []
    for min_val, max_val in zip(umin, umax):
        bounds += [(min_val, max_val)] * (m)

    # Define the dumax constraint
    def dumax_constraint(U_flatted, *args):
        U_ = U_flatted.reshape((nu, m))
        u0_ = u0temp.reshape(-1, 1)
        U_ = np.hstack((u0_, U_))
        du_ = np.diff(U_, axis=1)
        dumax_reshaped = np.tile(dumax, (m, 1)).T
        dumax_constraint = dumax_reshaped - np.abs(du_)
        return dumax_constraint.flatten()  # g(x) > =0

    constraints = [{'type': 'ineq', 'fun': dumax_constraint}]

    # Optimize the control objective function
    result = minimize(control_objective,
                      U_flatted,
                      bounds=bounds,
                      constraints=constraints,
                      args=args,
                      method='SLSQP')

    # Reshape the optimized U
    U_opt = result.x.reshape((nu, -1))

    # Return the first control action
    u_opt = U_opt[:, 0]
    end_time = time.time()  # End timing
    print(f'mpc controller calculation completed in {round(end_time - start_time, 2)} seconds')
    return u_opt
