import numpy as np
import control as ct


class System1:
    ''' represents a dynamic system'''

    def __init__(self):
        # Parameters
        self.nu = 3  # Number of manipulated inputs
        self.ny = 2  # Number of controlled outputs
        self.T = 1  # Sampling time (min)
        self.n = 300  # Stabilizing horizon
        # self.p = 30  # Output prediction horizon 1
        self.p = 200  # Output prediction horizon 2
        self.m = 3  # Control horizon 1
        # self.m = 2  # Control horizon 2

        # Initial input and output values
        self.u0 = np.array([700, 6.2, 0])
        # self.y0 = np.array([2.5, 0.7]) # inside the bounds
        # self.y0 = np.array([3,2, 0.7]) # out of bounds 1
        self.y0 = np.array([2.5, 0.5])  # out of bounds 2
        # self.y0 = np.array([3.2, 0.5])  # out of bounds 3

        # Input constraints
        self.umax = np.array([950, 9.0, 1e3])
        self.umin = np.array([400, 3.0, -1e3])
        self.dumax = np.array([10, 0.5, 1e-4])

        # Output constraints
        self.ymax = np.array([2.8, 0.75])
        self.ymin = np.array([2.0, 0.65])

        # Weights of the control layer
        # self.q = np.array([10000, 10000])  # Output weights
        self.q = np.array([100000, 100000])  # Output weights
        # self.r = np.array([100, 100, 1])  # Input weights
        self.r = np.array([10, 10, 1])  # Input weights

        # Weights of the economic layer
        self.py = np.array([0, 0])  # Output weights
        self.pu = np.array([0, 1, 0])  # Input weights
        self.peps = np.array([1e5, 1e5])  # Penalty weights
        self.ru = np.array([1, 1, 0])  # Input optimization weights

        # System Transfer functions
        self.G11 = ct.tf([-1.9973e-3, -1.3105e-4], [1, -8.3071e-1, -5.4544e-1, 5.1700e-1, 0],
                         dt=self.T)
        self.G21 = ct.tf([9.2722e-5, 3.1602e-5], [1, -1.1613, -7.5733e-2, 3.7749e-1, 0, 0, 0, 0],
                         dt=self.T)
        self.G12 = ct.tf([1.9486e-2 / 24, 4.6325e-2 / 24],
                         [1, -1.5119, 4.3596e-1, 8.2888e-2, 0, 0, 0, 0, 0, 0, 0],
                         dt=self.T)
        self.G22 = ct.tf([-4.2379e-2 / 24], [1, -1.3953, 2.5570e-1, 1.5459e-1, 0, 0, 0, 0, 0, 0, 0],
                         dt=self.T)
        self.G13 = ct.tf([-1.5789e-4, -8.3160e-5],
                         [1, -6.9794e-1, -1.8196e-1, -9.3550e-2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         dt=self.T)
        self.G23 = ct.tf([6.9654e-6, 8.6757e-6], [1, -8.8584e-1, 0, 0, 0, 0, 0, 0, 0, 0], dt=self.T)

        # Create a 2D list of transfer functions
        self.Gm = [[self.G11, self.G12, self.G13], [self.G21, self.G22, self.G23]]

    def generate_test_input(self, nsimTest):
        tsimTest = np.linspace(0, nsimTest * self.T, nsimTest + 1)  # Simulation Time vector
        Utest = np.tile(self.u0, (len(tsimTest), 1)).T  # Initialisation of the input matrix
        Utest[:, 100:] = np.array([710, 6.2, 0]).reshape(-1, 1)  # Step change in inputs
        Utest[:, 500:] = np.array([710, 6.3, 0]).reshape(-1, 1)  # Step change in inputs
        Utest[:, 1000:] = np.array([710, 6.3, 10]).reshape(-1, 1)  # Step change in inputs
        Utest[:, 1500:] = np.array([700, 6.2, 0]).reshape(-1, 1)  # Step change in inputs
        return Utest, tsimTest
