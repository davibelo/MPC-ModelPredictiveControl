import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def calculate_step_responses(Gm, nsim, T):
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
    PLOT_MAX_LENGHT = 500
    nrows = len(Gstep)
    ncols = len(Gstep[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
    fig.suptitle('step_responses')
    for j, row in enumerate(Gstep):
        for i, y_out in enumerate(row):
            axs[j, i].plot(tstep[:PLOT_MAX_LENGHT], y_out[:PLOT_MAX_LENGHT])
            axs[j, i].set_title(f'Gstep[{j}][{i}]')
            axs[j, i].grid(True)
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Step Response')
    plt.savefig('davi/figures/step_responses.png')

    return Gstep


def simulateMIMO(Gstep, tsim, ny, nu, y0, U):
    """
    Simulate the MIMO system response y given the step response matrix Gstep, time vector tsim,
    number of outputs ny, number of inputs nu, initial condition y0, and input matrix U, with the first
    element of U considered as u0 for each input and delta_U calculated accordingly.

    Parameters:
    - Gstep: The 3D array (or list that can be converted to an array) of the system's step response
             for each output to each input (dimensions [ny, nu, t]).
    - tsim: Time vector for the simulation (array).
    - ny: Number of outputs.
    - nu: Number of inputs.
    - y0: Initial condition of y (array with dimensions [ny]).
    - U: Matrix of input vectors at each time step (array with dimensions [nu, len(tsim)]), where the
         first element of each input vector is considered as the initial condition u0.

    Returns:
    - Y: The simulated response of the system (array with dimensions [ny, len(tsim)]).
    """
    Gstep = np.array(Gstep)
    tsim_len = len(tsim)
    Y = np.zeros((ny, tsim_len))
    delta_U = np.zeros_like(U)
    for t in range(tsim_len):
        for j in range(ny):
            y = y0[j]
            for i in range(nu):
                # Calculate delta_U as the difference from the previous time step
                delta_U[i, t] = U[i, t] - U[i, t - 1] if t > 0 else U[i, 0] - U[i, 0]
                # Perform convolution only if t > 0
                if t > 0:
                    y += np.sum(
                        np.convolve(Gstep[j, i, :t + 1],
                                    np.full(t + 1, delta_U[i, :t + 1]),
                                    mode='valid'))
                else:
                    y += delta_U[i, t] * Gstep[j, i, 0]
            Y[j, t] = y

    return Y

def plot_simulation_results(Ysim, U, tsim, fig_name):
    # Plot simulation results
    nrows = max(len(U), len(Ysim))
    fig, axs = plt.subplots(nrows, 2, figsize=(15, 15))
    fig.suptitle(fig_name)
    for i, y_out in enumerate(Ysim):
        axs[i, 0].plot(tsim, y_out)
        axs[i, 0].set_title(f'YsimTest[{i}]')
        axs[i, 0].grid(True)
    for i, u_out in enumerate(U):
        axs[i, 1].plot(tsim, u_out)
        axs[i, 1].set_title(f'U[{i}]')
        axs[i, 1].grid(True)
    for ax in axs[-1]:
        ax.set(xlabel='Time')
    for ax in axs[:, 0]:
        ax.set(ylabel='Ysim')
    for ax in axs[:, 1]:
        ax.set(ylabel='U')
    plt.savefig(f'davi/figures/{fig_name}.png')


def mpc_controller(ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r, U0, Y0, Gstep):
    # Define the control objective function
    def control_objective(U, *args):
        ''' Control objective function '''
        ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r, U0, Y0, Gstep = args
        J = 0
        tp = np.linspace(0, (p - 1) * T, p)  # Time vector
        # Reshape U back to its original shape
        U = U.reshape((nu, -1))
        print(U)
        # Only consider the first m values of U and keep the rest constant
        U = np.hstack((U[:, :m], np.repeat(U[:, m - 1:], p - m, axis=1)))

        yp = simulateMIMO(Gstep, tp, ny, nu, Y0, U)
        ymin = np.asarray(ymin).reshape(-1, 1)
        ymax = np.asarray(ymax).reshape(-1, 1)
        q = np.asarray(q).reshape(-1, 1)

        assert yp.ndim == 2 and yp.shape[0] == ymin.shape[0]

        error = np.sum(q * ((yp < ymin) * (ymin - yp) + (yp > ymax) * (yp - ymax)))
        J += error
        # delta_u = np.diff(u, axis=1)
        # J += np.sum(r * np.sum(delta_u**2, axis=1))

        return J

    # Initialize U as U0 on all control moves
    U = np.tile(U0.reshape(-1, 1), p)
    # print(U)
    # Define the bounds for U
    bounds = [(umin[i % nu], umax[i % nu]) for i in range(nu * p)]

    # Define the arguments for the control objective function
    args = (ny, nu, T, n, p, m, umax, umin, ymax, ymin, dumax, q, r, U0, Y0, Gstep)

    # Optimize the control objective function
    U_flatted = U.flatten()
    print(U_flatted)
    result = minimize(control_objective, U_flatted, args=args, bounds=bounds, method='SLSQP')

    # Reshape the optimized U
    U_opt = result.x.reshape((nu, -1))

    # Return the first control action
    return U_opt[:, 0]
