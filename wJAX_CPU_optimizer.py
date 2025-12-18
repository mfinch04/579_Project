# Imports
import numpy as onp # numpy for vectorization
from collections.abc import Callable # For type hints
import matplotlib.pyplot as plt
from scipy import optimize
import jax
import jax.numpy as np
class HeatEquation2D:
    """Heat Equation Solver for MECH 579 Final Project

    This class will construct and solve the unsteady heat equation
    with Robin BCs as described in the assignment.
    """
    def __init__(self, x:float, y:float, height:float , n_x:int, n_y:int,
                     k:float=1.0, rho:float=1.0, cp:float=1.0,
                     CFL:float=0.1, init_condition:Callable[[np.ndarray,np.ndarray], np.ndarray] = lambda x,y: np.sin(x+y)):
        """Intializition function for the heat equation

        Parameters

        ------

        x (float): Physical Size of CPU in x-direction [m]

        y (float): Physical Size of CPU in y-direction [m]

        n_x (int): Number of grid points in x-direction [m]

        n_y (int): Number of grid points in y-direction [m]

        k (float): The heat transfer coefficient of the CPU [W/[mK]]

        rho (float): Constant density of CPU [kg/m^3]

        cp (float): Specific heat capacity of CPU [kJ/[kgK]]

        CFL (float): Courant-Friedrichs-Lewy Number

        init_condition (function(x,y)): Initial condition of the CPU
        """
        ## MESHING variables
        self.n_x = n_x
        self.n_y = n_y
        self.boundary_conditions = []
        # Physical locations
        x_axis = np.linspace(0, x, self.n_x)
        y_axis = np.linspace(0, y, self.n_y)
        self.X, self.Y = np.meshgrid(x_axis, y_axis, indexing='ij')
        self.dx = x_axis[1] - x_axis[0]
        self.dy = y_axis[1] - y_axis[0]
        # Variables of Mesh size
        self.u = np.zeros((self.n_x, self.n_y))
        self.h_top_values = np.zeros((self.n_x, self.n_y))
        self.h_boundary_values = np.zeros((self.n_x, self.n_y))

        ## Heat Generation Properties
        self.heat_generation_function = lambda x, y, a, b, c: a * x + b * y + c  # Can be changed
        self.heat_gen_a = 0
        self.heat_gen_b = 0
        self.heat_gen_c = 0
        self.heat_generation_total = 0

        ## Material Properties
        self.k = k
        self.rho = rho
        self.cp = cp
        self.thermal_alpha = self.k / (self.rho * self.cp)
        self.height = height #m

        ## Temporal Properties
        self.CFL = CFL
        self.dt = self.CFL * (self.dx * self.dy) / self.thermal_alpha
        self.current_time = 0
        self.steady_state_error = 1E2 # Large inital number to ensure that the problem will continue
        self.max_iter = 5E4
        self.init_condition = init_condition
        self.apply_initial_conditions()

        ## External Variables of Air
        self.ext_k = 0.02772  # W/m/K Thermal Coeffcient
        self.ext_Pr = 0.7215  # Prantl Number
        self.ext_nu = 1.506 * 10 ** (-5)  # m^2/s Kinematic Viscosity
        self.ext_T = 273 + 20  # K Temperature

        ## Fan Variables
        self.v = 10 # m/s Air Velocity
        self.fan_efficiency_func = lambda v: -0.002*v**2 + 0.08*v
        self.fan_efficiency = self.fan_efficiency_func(self.v)

        self.verbose = False

    def set_initial_conditions(self,initial_conditions:Callable[[np.ndarray,np.ndarray],np.ndarray]):
        """Sets the initial condition

        Parameters

        ------

        initial_conditions(function(x,y)): Initial condition of the CPU
        """
        self.init_condition = initial_conditions

    def apply_initial_conditions(self):
        """Applies the initial condition into self.u"""
        self.u = self.init_condition(self.X,self.Y)

    def reset(self):
        """Resets the heat equation"""
        self.apply_initial_conditions()
        self.current_time = 0
        self.steady_state_error = 1E2

    def set_heat_generation(self, heat_generation_function: Callable[[np.ndarray,np.ndarray,float,float,float], np.ndarray],
                            a: float, b: float, c: float):
        """Sets the heat generation function and associated variables

        Parameters

        ------

        heat_generation_function (function(x,y,a,b,c)): Function that dictates the heat generation by the CPU

        integrated_total (float): Total integrated value

        a, b, c (float): Variables associated with the heat generation function
        """

        q = heat_generation_function(self.X, self.Y, a, b, c)   # must use jnp ops inside
        w = np.ones_like(q)
        w = w.at[0, :].multiply(0.5); w = w.at[-1, :].multiply(0.5)
        w = w.at[:, 0].multiply(0.5); w = w.at[:, -1].multiply(0.5)
        
        total = np.sum(q * w) * self.dx * self.dy * self.height
    
        # # edges (halve)
        # heat_generation_matrix = heat_generation_matrix.at[i0, :].set(heat_generation_matrix[i0, :] / 2)
        # heat_generation_matrix = heat_generation_matrix.at[iN, :].set(heat_generation_matrix[iN, :] / 2)
        # heat_generation_matrix = heat_generation_matrix.at[:, j0].set(heat_generation_matrix[:, j0] / 2)
        # heat_generation_matrix = heat_generation_matrix.at[:, jN].set(heat_generation_matrix[:, jN] / 2)

        # # corners (halve again -> net quarter)
        # heat_generation_matrix = heat_generation_matrix.at[i0, j0].set(heat_generation_matrix[i0, j0] / 2)
        # heat_generation_matrix = heat_generation_matrix.at[i0, jN].set(heat_generation_matrix[i0, jN] / 2)
        # heat_generation_matrix = heat_generation_matrix.at[iN, j0].set(heat_generation_matrix[iN, j0] / 2)
        # heat_generation_matrix = heat_generation_matrix.at[iN, jN].set(heat_generation_matrix[iN, jN] / 2)

    def set_fan_velocity(self, v: float):
        """Sets the fan velocity

        Parameters

        ------

        v (float): Variable associated with the fan velocity
        """
        self.v = v
        self.fan_efficiency = self.fan_efficiency_func(self.v)


    # def h_boundary(self,u: np.ndarray):
    #     """Calculates the convective heat transfer coefficient at the boundaries

    #     Parameters

    #     ------

    #     u (np.ndarray): Current Temperature Mesh
    #     """
    #     beta = 1/((u+self.ext_T)/2)
    #     rayleigh = 9.81*beta*(u-self.ext_T)*self.dx**3/(self.ext_nu**2)*self.ext_Pr
    #     nusselt = (0.825 + (0.387*rayleigh**(1/6))/
    #                (1+(0.492/self.ext_Pr)**(9/16))**(8/27))**2
    #     return nusselt*self.ext_k/self.dx
    
    def h_boundary(self, u):
        Tfilm = (u + self.ext_T) / 2.0
        Tfilm = np.maximum(Tfilm, 1e-6)      # avoid divide-by-zero

        beta = 1.0 / Tfilm

        dT = np.sqrt((u - self.ext_T)**2 + 1e-12)  # smooth appxroximation
        rayleigh = 9.81 * beta * dT * (self.dx**3) / (self.ext_nu**2) * self.ext_Pr
        rayleigh = np.maximum(rayleigh, 1e-12) # safety

        nusselt = (0.825 + (0.387 * rayleigh**(1.0/6.0)) /
                (1.0 + (0.492/self.ext_Pr)**(9.0/16.0))**(8.0/27.0))**2

        return nusselt * self.ext_k / self.dx

    def h_top(self, x: np.ndarray, v):
        """Calculates the convective heat transfer coefficient from the fan velocity
        
        Parameters

        ------

        x (np.ndarray): x position

        u (np.ndarray): UNUSED
        """

        ## converted to jax friendly format

        # Rex = v * x / self.ext_nu

        # Pr_term = self.ext_Pr ** (1.0 / 3.0)

        # # laminar and turbulent correlations (vectorized)
        # Nu_lam = 0.332 * np.power(Rex, 0.5) * Pr_term
        # Nu_tur = 0.0296 * np.power(Rex, 0.8) * Pr_term

        # Nux = np.where(Rex < 5e5, Nu_lam, Nu_tur)

        # h = Nux * self.ext_k / (x + 1e-5)
        # return h
        
        # pick an epsilon based on grid spacing (better than a random 1e-12)
        # if eps_x is None:
        #     eps_x = 0.5 * self.dx  # or self.dx, or 1e-6

        x_safe = np.maximum(x, 1e-6)             # ensures x_safe > 0 everywhere
        Rex = v * x_safe / self.ext_nu

        Pr_term = self.ext_Pr ** (1.0 / 3.0)

        Nu_lam = 0.332 * np.sqrt(Rex) * Pr_term   # sqrt is clearer than power(.,0.5)
        Nu_tur = 0.0296 * np.power(Rex, 0.8) * Pr_term

        Nux = np.where(Rex < 5e5, Nu_lam, Nu_tur)

        h = Nux * self.ext_k / (x_safe + 1e-5)    # use x_safe here too
        return h

        # Rex = self.v*x/self.ext_nu
        # r,c = Rex.shape
        # Nux = np.zeros((r,c))
        # for i in range(r):
        #     for j in range(c):
        #         if Rex[i,j] < 5E5:
        #             Nux[i,j] = 0.332*Rex[i,j]**0.5*self.ext_Pr**(1/3)
        #         else:
        #             Nux[i,j] = 0.0296*Rex[i,j]**0.8*self.ext_Pr**(1/3)
        # h = Nux*self.ext_k/(x + 1E-5)
        # return h

        # return h_top_jax(x, v, self.ext_nu, self.ext_Pr, self.ext_k)

    def calculate_h(self):
        """Calculates all necessary convective heat transfer coefficients"""
        self.h_top_values = self.h_top(self.X, self.v)
        self.h_boundary_values = self.h_boundary(self.u)

    def apply_boundary_conditions(self, old_u):
        """Calculates the change in temperature at the boundary.

        Parameters

        -----

        old_u (np.ndarray): Current Temperature Mesh
        """
        e_dot = self.heat_generation_function(self.X, self.Y, self.heat_gen_a, self.heat_gen_b, self.heat_gen_c)
        tau = self.thermal_alpha * self.dt / (self.dx*self.dy)
        i0,j0,iN,jN = 0, 0, self.n_x-1, self.n_y-1

        new_u = old_u.copy()
        # Left
        left_new = (old_u[i0,1:-1] +
                            2 * tau * self.h_boundary_values[i0,1:-1]/self.k * self.dy * (self.ext_T - old_u[i0,1:-1]) +
                            tau * self.dx * (old_u[i0,2:] - old_u[i0,1:-1]) / self.dy +
                            tau * self.dx * (old_u[i0,1:-1] - old_u[i0,2:]) / self.dy +
                            2 * tau * self.dy * (old_u[i0 + 1, 1:-1] - old_u[i0, 1:-1]) / self.dx +
                            tau * self.h_top_values[i0,1:-1]/self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[i0,1:-1]) +
                            tau * e_dot[i0,1:-1] / self.k * self.dx * self.dy)
        new_u = new_u.at[i0,1:-1].set(left_new)

        # Right
        right_new = (old_u[iN, 1:-1] +
                            2 * tau * self.h_boundary_values[iN, 1:-1] / self.k * self.dy * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * self.dx * (old_u[iN, 2:] - old_u[iN, 1:-1]) / self.dy +
                            tau * self.dx * (old_u[iN, 1:-1] - old_u[iN, 2:]) / self.dy +
                            2 * tau * self.dy * (old_u[iN- 1, 1:-1] - old_u[iN,1:-1]) / self.dx +
                            tau * self.h_top_values[iN, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * e_dot[iN, 1:-1] / self.k * self.dx * self.dy)
        new_u = new_u.at[iN, 1:-1].set(right_new)

        # Bottom
        bottom_new = (old_u[1:-1,j0] +
                            2 * tau * self.h_boundary_values[1:-1,j0] / self.k * self.dx * (self.ext_T - old_u[1:-1,j0]) +
                            tau * self.dy * (old_u[2:,j0] - old_u[1:-1,j0]) / self.dx +
                            tau * self.dy * (old_u[1:-1,j0] - old_u[2:,j0]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1,j0 + 1] - old_u[1:-1,j0]) / self.dy +
                            tau * self.h_top_values[1:-1,j0] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[1:-1,j0]) +
                            tau * e_dot[1:-1,j0] / self.k * self.dx * self.dy)
        new_u = new_u.at[1:-1,j0].set(bottom_new)

        # Top
        top_new = (old_u[1:-1,jN] +
                            2 * tau * self.h_boundary_values[1:-1,jN] / self.k * self.dx * (self.ext_T - old_u[1:-1,jN]) +
                            tau * self.dy * (old_u[2:,jN] - old_u[1:-1,jN]) / self.dx +
                            tau * self.dy * (old_u[1:-1,jN] - old_u[2:,jN]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1,jN - 1] - old_u[1:-1,jN]) / self.dy +
                            tau * self.h_top_values[1:-1,jN] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[1:-1,jN]) +
                            tau * e_dot[1:-1, jN] / self.k * self.dx * self.dy)
        new_u = new_u.at[1:-1,jN].set(top_new)

        ## Bottom Left Corner
        botton_left_new = (old_u[i0,j0] +
                         2 * tau * self.h_boundary_values[i0,j0] * self.dy / self.k * (self.ext_T - old_u[i0,j0]) +
                         2 * tau * self.h_boundary_values[i0,j0] * self.dx / self.k * (self.ext_T - old_u[i0,j0]) +
                         2 * tau * self.dx * (old_u[i0,j0+1] - old_u[i0,j0]) / self.dy +
                         2 * tau * self.dy * (old_u[i0+1,j0] - old_u[i0,j0]) / self.dx +
                         tau * self.h_top_values[i0,j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[i0,j0]) +
                         tau * e_dot[i0,j0] / self.k * self.dx * self.dy)
        new_u = new_u.at[i0,j0].set(botton_left_new)

        ## Bottom Right Corner
        bottom_right_new = (old_u[iN,j0] +
                         2 * tau * self.h_boundary_values[iN,j0] * self.dy / self.k * (self.ext_T - old_u[iN,j0]) +
                         2 * tau * self.h_boundary_values[iN,j0] * self.dx / self.k * (self.ext_T - old_u[iN,j0]) +
                         2 * tau * self.dx * (old_u[iN,j0+1] - old_u[iN,j0]) / self.dy +
                         2 * tau * self.dy * (old_u[iN-1,j0] - old_u[iN,j0]) / self.dx +
                         tau * self.h_top_values[iN,j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN,j0]) +
                         tau * e_dot[iN,j0] / self.k * self.dx * self.dy)
        new_u = new_u.at[iN,j0].set(bottom_right_new)

        ## Top Left Corner
        top_left_new = (old_u[i0,jN] +
                         2 * tau * self.h_boundary_values[i0,jN] * self.dy / self.k * (self.ext_T - old_u[i0,jN]) +
                         2 * tau * self.h_boundary_values[i0,jN] * self.dx / self.k * (self.ext_T - old_u[i0,jN]) +
                         2 * tau * self.dx * (old_u[i0,jN-1] - old_u[i0,jN]) / self.dy +
                         2 * tau * self.dy * (old_u[i0+1,jN] - old_u[i0,jN]) / self.dx +
                         tau * self.h_top_values[i0,jN] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[i0,jN]) +
                         tau * e_dot[i0,jN] / self.k * self.dx * self.dy)
        new_u = new_u.at[i0,jN].set(top_left_new)

        ## Top Right Corner
        top_right_new = (old_u[iN,jN] +
                         2 * tau * self.h_boundary_values[iN,jN] * self.dy / self.k * (self.ext_T - old_u[iN,jN]) +
                         2 * tau * self.h_boundary_values[iN,jN] * self.dx / self.k * (self.ext_T - old_u[iN,jN]) +
                         2 * tau * self.dx * (old_u[iN,jN-1] - old_u[iN,jN]) / self.dy +
                         2 * tau * self.dy * (old_u[iN-1,jN] - old_u[iN,jN]) / self.dx +
                         tau * self.h_top_values[iN,jN] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN,jN]) +
                         tau * e_dot[iN,jN] / self.k * self.dx * self.dy)
        new_u = new_u.at[iN,jN].set(top_right_new)

        return new_u
    @staticmethod
    def apply_boundary_conditions_pure(
        old_u,          # (nx, ny)
        e_dot,          # (nx, ny)
        h_bnd,          # (nx, ny) = h_boundary(old_u)
        h_top,          # (nx, ny) = h_top(X, v)
        tau,            # scalar
        k, dx, dy,      # scalars
        height,         # scalar
        ext_T           # scalar (ambient temp)
    ):
        nx, ny = old_u.shape
        i0, iN = 0, nx - 1
        j0, jN = 0, ny - 1

        # Start from old_u so interior stays unchanged
        u = old_u

        # -------------------------
        # Left boundary (i = 0), j = 1..ny-2
        # -------------------------
        left = (
            old_u[i0, 1:-1]
            + 2 * tau * h_bnd[i0, 1:-1] / k * dy * (ext_T - old_u[i0, 1:-1])
            + tau * dx * (old_u[i0, 2:]   - old_u[i0, 1:-1]) / dy
            + tau * dx * (old_u[i0, 0:-2] - old_u[i0, 1:-1]) / dy
            + 2 * tau * dy * (old_u[i0 + 1, 1:-1] - old_u[i0, 1:-1]) / dx
            + tau * h_top[i0, 1:-1] / k * dx * dy / height * (ext_T - old_u[i0, 1:-1])
            + tau * e_dot[i0, 1:-1] / k * dx * dy
        )
        u = u.at[i0, 1:-1].set(left)

        # -------------------------
        # Right boundary (i = nx-1)
        # -------------------------
        right = (
            old_u[iN, 1:-1]
            + 2 * tau * h_bnd[iN, 1:-1] / k * dy * (ext_T - old_u[iN, 1:-1])
            + tau * dx * (old_u[iN, 2:]   - old_u[iN, 1:-1]) / dy
            + tau * dx * (old_u[iN, 0:-2] - old_u[iN, 1:-1]) / dy
            + 2 * tau * dy * (old_u[iN - 1, 1:-1] - old_u[iN, 1:-1]) / dx
            + tau * h_top[iN, 1:-1] / k * dx * dy / height * (ext_T - old_u[iN, 1:-1])
            + tau * e_dot[iN, 1:-1] / k * dx * dy
        )
        u = u.at[iN, 1:-1].set(right)

        # -------------------------
        # Bottom boundary (j = 0), i = 1..nx-2
        # -------------------------
        bottom = (
            old_u[1:-1, j0]
            + 2 * tau * h_bnd[1:-1, j0] / k * dx * (ext_T - old_u[1:-1, j0])
            + tau * dy * (old_u[2:,   j0] - old_u[1:-1, j0]) / dx
            + tau * dy * (old_u[0:-2, j0] - old_u[1:-1, j0]) / dx
            + 2 * tau * dx * (old_u[1:-1, j0 + 1] - old_u[1:-1, j0]) / dy
            + tau * h_top[1:-1, j0] / k * dx * dy / height * (ext_T - old_u[1:-1, j0])
            + tau * e_dot[1:-1, j0] / k * dx * dy
        )
        u = u.at[1:-1, j0].set(bottom)

        # -------------------------
        # Top boundary (j = ny-1)
        # -------------------------
        top = (
            old_u[1:-1, jN]
            + 2 * tau * h_bnd[1:-1, jN] / k * dx * (ext_T - old_u[1:-1, jN])
            + tau * dy * (old_u[2:,   jN] - old_u[1:-1, jN]) / dx
            + tau * dy * (old_u[0:-2, jN] - old_u[1:-1, jN]) / dx
            + 2 * tau * dx * (old_u[1:-1, jN - 1] - old_u[1:-1, jN]) / dy
            + tau * h_top[1:-1, jN] / k * dx * dy / height * (ext_T - old_u[1:-1, jN])
            + tau * e_dot[1:-1, jN] / k * dx * dy
        )
        u = u.at[1:-1, jN].set(top)

        # -------------------------
        # Corners
        # -------------------------
        # Bottom-left (0,0)
        bl = (
            old_u[i0, j0]
            + 2 * tau * h_bnd[i0, j0] * dy / k * (ext_T - old_u[i0, j0])
            + 2 * tau * h_bnd[i0, j0] * dx / k * (ext_T - old_u[i0, j0])
            + 2 * tau * dx * (old_u[i0, j0 + 1] - old_u[i0, j0]) / dy
            + 2 * tau * dy * (old_u[i0 + 1, j0] - old_u[i0, j0]) / dx
            + tau * h_top[i0, j0] / k * dx * dy / height * (ext_T - old_u[i0, j0])
            + tau * e_dot[i0, j0] / k * dx * dy
        )
        u = u.at[i0, j0].set(bl)

        # Bottom-right (nx-1,0)
        br = (
            old_u[iN, j0]
            + 2 * tau * h_bnd[iN, j0] * dy / k * (ext_T - old_u[iN, j0])
            + 2 * tau * h_bnd[iN, j0] * dx / k * (ext_T - old_u[iN, j0])
            + 2 * tau * dx * (old_u[iN, j0 + 1] - old_u[iN, j0]) / dy
            + 2 * tau * dy * (old_u[iN - 1, j0] - old_u[iN, j0]) / dx
            + tau * h_top[iN, j0] / k * dx * dy / height * (ext_T - old_u[iN, j0])
            + tau * e_dot[iN, j0] / k * dx * dy
        )
        u = u.at[iN, j0].set(br)

        # Top-left (0,ny-1)
        tl = (
            old_u[i0, jN]
            + 2 * tau * h_bnd[i0, jN] * dy / k * (ext_T - old_u[i0, jN])
            + 2 * tau * h_bnd[i0, jN] * dx / k * (ext_T - old_u[i0, jN])
            + 2 * tau * dx * (old_u[i0, jN - 1] - old_u[i0, jN]) / dy
            + 2 * tau * dy * (old_u[i0 + 1, jN] - old_u[i0, jN]) / dx
            + tau * h_top[i0, jN] / k * dx * dy / height * (ext_T - old_u[i0, jN])
            + tau * e_dot[i0, jN] / k * dx * dy
        )
        u = u.at[i0, jN].set(tl)

        # Top-right (nx-1,ny-1)
        tr = (
            old_u[iN, jN]
            + 2 * tau * h_bnd[iN, jN] * dy / k * (ext_T - old_u[iN, jN])
            + 2 * tau * h_bnd[iN, jN] * dx / k * (ext_T - old_u[iN, jN])
            + 2 * tau * dx * (old_u[iN, jN - 1] - old_u[iN, jN]) / dy
            + 2 * tau * dy * (old_u[iN - 1, jN] - old_u[iN, jN]) / dx
            + tau * h_top[iN, jN] / k * dx * dy / height * (ext_T - old_u[iN, jN])
            + tau * e_dot[iN, jN] / k * dx * dy
        )
        u = u.at[iN, jN].set(tr)

        return u

    @staticmethod
    def interior_update_pure(old_u, h_top, X, Y, a, b, c,
                         tau, k, dx, dy, height, ext_T, heat_fun):
        lap = (
            dy * (old_u[2:, 1:-1] - 2*old_u[1:-1, 1:-1] + old_u[0:-2, 1:-1]) / dx
            + dx * (old_u[1:-1, 2:] - 2*old_u[1:-1, 1:-1] + old_u[1:-1, 0:-2]) / dy
        )

        src = heat_fun(X[1:-1, 1:-1], Y[1:-1, 1:-1], a, b, c)

        interior = (
            old_u[1:-1, 1:-1]
            + tau * lap
            + tau * (
                h_top[1:-1, 1:-1] / k * dx * dy / height * (ext_T - old_u[1:-1, 1:-1])
                + dx * dy / k * src
            )
        )

        u_next = old_u.at[1:-1, 1:-1].set(interior)
        return u_next
    
    def step_forward_in_time(self):
        """Steps forward in time 1 timestep"""
        self.h_top_values = self.h_top(self.X, self.v)
        self.h_boundary_values = self.h_boundary(self.u)
        old_u = self.u.copy()
        self.u = self.apply_boundary_conditions(old_u)
        tau = self.thermal_alpha * self.dt / (self.dx * self.dy)
        # internal node update
        self.u = self.u.at[1:-1, 1:-1].set((old_u[1:-1, 1:-1] +
                                    tau * (
                                            self.dy * (old_u[2:, 1:-1] - 2 * old_u[1:-1, 1:-1] + old_u[0:-2, 1:-1]) / self.dx  +
                                            self.dx * (old_u[1:-1, 2:] - 2 * old_u[1:-1, 1:-1] + old_u[1:-1, 0:-2]) / self.dy
                                    ) + tau * (self.h_top_values[1:-1, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[1:-1, 1:-1]) +
                                    self.dx * self.dy / self.k * self.heat_generation_function(self.X[1:-1, 1:-1],self.Y[1:-1, 1:-1],self.heat_gen_a,self.heat_gen_b,self.heat_gen_c))))
        self.steady_state_error = np.linalg.norm(self.u - old_u,np.inf)
        self.current_time += self.dt

    def solve_until_steady_state(self, tol: float = 1e-3):
        """Solves until steady state is reached

        Parameters

        ------

        tol (float, optional): Tolerance until steady state
        """
        iter = 0
        self.step_forward_in_time()
        while self.steady_state_error > tol and iter < self.max_iter:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Error: {self.steady_state_error}")


    def solve_until_time(self,final_time: float):
        """Solves until time is reached

        Parameters

        ------

        final_time (float): Final time of simulation
        """
        iter = 0
        while self.current_time < final_time:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Time: {self.current_time}")


# if __name__ == "__main__":
#     # Physical Dimensions
#     cpu_x = 0.04  # m
#     cpu_y = 0.04  # m
#     cpu_z = 0.04  # m
#     N = 25

#     # Temporal Parameters
#     CFL = 0.5
#     # Silicon Constants
#     k_si = 149
#     rho_si = 2323
#     c_si = 19.789 / 28.085 * 1000  # J/(kgK)


#     def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#         r, c = x.shape
#         u = np.zeros([r, c])
#         ## Cosine Case
#         u = 70 * np.sin(x * np.pi / cpu_x) * np.sin(y * np.pi / cpu_y) + 293
#         return u


#     def heat_generation_function(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
#         return a * x + b * y + c
    
    
#     ## Problem Set up
#     heq = HeatEquation2D(cpu_x,cpu_y,cpu_z, N,N,
#                        k=k_si,rho=rho_si,cp=c_si,
#                        init_condition=initial_condition)
#     """
#     # Test values for a,b,c
#     test_a = 1*10**6
#     test_b = 1*10**6
#     test_c = (1.5625*10**5 - 0.02*test_b - 0.02*test_a)
#     ## Fan velocity for test
#     fan_velocity = 10.0
#     heq.set_heat_generation(heat_generation_function,test_a,test_b,test_c)
#     heq.set_fan_velocity(fan_velocity)
#     ## plotting initial conditions
#     fig, ax = plt.subplots()
#     contour1 = ax.contourf(heq.X,heq.Y,heq.u - 273)
#     fig.colorbar(contour1,ax=ax)
#     #plt.show()
# """

#     ## Setting objective function
#     heq.max_iter = 5E5
#     w1 = 0.2
#     w2 = 1 - w1
#     global_tolerance = 1E-3
    
#     x_vals = []
#     obj_vals = []
#     grad_vals = []
#     gradL_vals = []
#     power_vals = []
#     T_vals = []
#     eta_vals = []
#     eps_vals = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4] # x-axis of differential plots


#     def objective_function(x):
#         # x = [v, a, b, c]


#         heq.set_fan_velocity(x[0])
#         heq.set_heat_generation(heat_generation_function, x[1], x[2], x[3])

#         #Reset time and temperature field
#         heq.reset()

#         #Solve PDE
#         heq.solve_until_steady_state(tol=global_tolerance)

#         #Compute objective 
#         Tmax = np.max(heq.u)
#         fan_eff = heq.fan_efficiency

#         obj =  w1 * (Tmax / 273) - w2 * fan_eff

#         x_vals.append(x)
#         T_vals.append(Tmax)
#         eta_vals.append(fan_eff)
#         obj_vals.append(obj)
#         power_vals.append(heq.heat_generation_total)

#         return obj


#     ## Bounds for inputs
#     bounds = [
#         (0, 30),
#         (-np.inf, np.inf),
#         (-np.inf, np.inf),
#         (0, np.inf),
#     ]

#     def constraint_one(x):
#         heq.set_heat_generation(heat_generation_function, x[1], x[2], x[3])
#         return 10 - heq.heat_generation_total
    
#     def callback_func(xk, state):

#         if hasattr(state, 'lagrangian_grad'):
#             grad_L = state.lagrangian_grad
#             grad_L_norm = np.linalg.norm(grad_L)
#             gradL_vals.append(grad_L_norm)
#         else:
#             # Handle cases where the attribute might not exist (e.g., other methods or versions)
#             print(f"Warning: callback did not provide 'lagrangian_grad'")


#     ## Setting the constraints
#     constraints = [
#         {'type': 'eq', 'fun': constraint_one},
#     ]
#     ## Creating the initial guess
#     v0 = 10
#     x0_heat = 0
#     x0 = [v0, x0_heat * 10 ** 5, x0_heat * 10 ** 5, (156250 - 0.02 * x0_heat * 10 ** 5 - 0.02 * x0_heat * 10 ** 5)]
#     heq.verbose = False
#     ## Optimize
#     optimization_result = optimize.minimize(
#         objective_function,
#         x0,
#         method='trust-constr',
#         bounds=bounds,
#         constraints=constraints, 
#         callback=callback_func,
#         options={'verbose': 3}
#     )
#     ## Build optimal solution
#     heq.set_fan_velocity(optimization_result.x[0])
#     heq.set_heat_generation(heat_generation_function, optimization_result.x[1], optimization_result.x[2],
#                             optimization_result.x[3])
#     print(
#         f"Optimization result: {objective_function(optimization_result.x)}\n"
#         f"v: {optimization_result.x[0]} m/s, "
#         f"a: {optimization_result.x[1]}, "
#         f"b: {optimization_result.x[2]}, "
#         f"c: {optimization_result.x[3]}"
#         f"\n"
#         f"Constraints:\n"
#         f"Total Heat Generation: {heq.heat_generation_total} Constraint: {constraint_one(optimization_result.x)}\n"
#     )
#     ## Plot optimal solution
#     fig, ax = plt.subplots()
#     contour3 = ax.contourf(heq.X, heq.Y, heq.u - 273)
#     fig.colorbar(contour3, ax=ax)
#     plt.show()


#     iterations = np.arange(len(obj_vals))
#     obj_list = [float(v) for v in obj_vals]

#     plt.figure(figsize=(12,8))
#     plt.plot(iterations, obj_list, label="Objective")
#     print(type(obj_vals[0]))
#     print(type(obj_list[0]))
#     print("obj:", obj_list)
#     plt.plot(iterations, T_vals, label="Max Temperature")
#     plt.plot(iterations, eta_vals, label="Fan Efficiency")
#     plt.plot(iterations, power_vals, label="Constraint Value")
#     plt.yscale('symlog')
#     plt.legend()
#     plt.xlabel("Iteration")
#     plt.ylabel("Value")
#     plt.title("Convergence History")
#     plt.grid(True)
#     plt.show()

#     x_array = np.array(x_vals)
#     x_array = x_array.astype(float)

#     plt.figure(figsize=(12,8))
#     plt.plot(iterations, x_array[:,0], label="v (fan)")
#     plt.plot(iterations, x_array[:,1], label="a")
#     print("a:", x_array[:,1] )
#     print("b:", x_array[:,2] )
#     plt.plot(iterations, x_array[:,2], label="b")
#     plt.plot(iterations, x_array[:,3], label="c")
#     plt.yscale('symlog')
#     plt.legend()
#     plt.title("Parameter Convergence")
#     plt.grid(True)
#     plt.show()


#     grad_iterations = np.arange(len(gradL_vals))
#     plt.figure(figsize=(12,8))
#     print("gradL_vals:", gradL_vals)
#     plt.plot(grad_iterations, gradL_vals)
#     plt.xlabel("Iteration")
#     plt.ylabel("||grad L||")
#     plt.yscale('log')
#     #plt.legend()
#     plt.title("Lagrangian Convergence")
#     plt.grid(True)
#     plt.show()

# for part (c), obtaining plot of central difference finite difference gradient based on step sizes

if __name__ == "__main__":
    # Physical Dimensions
    cpu_x = 0.04  # m
    cpu_y = 0.04  # m
    cpu_z = 0.04  # m
    N = 10

    # Temporal Parameters
    CFL = 0.5
    # Silicon Constants
    k_si = 149
    rho_si = 2323
    c_si = 19.789 / 28.085 * 1000  # J/(kgK)


    def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r, c = x.shape
        u = np.zeros([r, c])
        ## Cosine Case
        u = 70 * np.sin(x * np.pi / cpu_x) * np.sin(y * np.pi / cpu_y) + 293
        return u


    def heat_generation_function(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x + b * y + c
    
    
    ## Problem Set up
    heq = HeatEquation2D(cpu_x,cpu_y,cpu_z, N,N,
                       k=k_si,rho=rho_si,cp=c_si,
                       init_condition=initial_condition)
        

    ## Setting objective function
    heq.max_iter = 5E5
    w1 = 0.2
    w2 = 1 - w1
    global_tolerance = 1E-3
    
    x_vals = []
    obj_vals = []
    grad_vals = []
    gradL_vals = []
    power_vals = []
    T_vals = []
    eta_vals = []
    eps_vals = np.geomspace(1e-5, 1e-1, 50)
    # eps_vals = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4] # x-axis of differential plots
    # eps_vals = [1e-1, 5e-2]
    FD_grad_J = []

    # def total_heat(a, b, c):
    #     q = heat_generation_function(heq.X, heq.Y, a, b, c)   # must use jnp ops inside
    #     w = np.ones_like(q)
    #     w = w.at[0, :].multiply(0.5); w = w.at[-1, :].multiply(0.5)
    #     w = w.at[:, 0].multiply(0.5); w = w.at[:, -1].multiply(0.5)
    #     return np.sum(q * w) * heq.dx * heq.dy * heq.height

    # g = jax.grad(total_heat, argnums=(0,1,2))(1.0, 2.0, 3.0)
    # print(g)

    # print("Testing JIT")
    # jit_total_heat = jax.jit(total_heat)
    # print(jit_total_heat(1.0, 2.0, 3.0))

    # def fd_a(a, b, c, h=1e-6):
    #     return (total_heat(a+h, b, c) - total_heat(a-h, b, c)) / (2*h)

    # a,b,c = 1.0,2.0,3.0
    # ga = float(jax.grad(total_heat, argnums=0)(a,b,c))
    # fa = float(fd_a(a,b,c,1e-1))
    # print("jax grad:", ga, ", FD grad:", fa,", diff:", abs(ga-fa))

    # print("Testing h_top")
    # X = np.ones((20, 20)) * 0.02  # example grid (must be >0)
    # v0 = 1.0

    # def scalar_test(v):
    #     return np.sum(heq.h_top(X, v))

    # dv = jax.grad(scalar_test)(v0)
    # print("dv=", dv)

    # print("Testing scalar_test jit \n")
    # scalar_test_jit = jax.jit(scalar_test)
    # print(scalar_test_jit(v0))
    # print(jax.grad(scalar_test_jit)(v0))

    # print("Comparing with FD \n")
    # def fd(v, h=1e-3):
    #     return (scalar_test(v+h) - scalar_test(v-h)) / (2*h)

    # print("AD:", float(jax.grad(scalar_test)(v0)))
    # for h in np.geomspace(1e-5, 1e-3, 10):
    #     print(f"FD, h={h}:", float(fd(v0, h=h)))

    # print("Testing boundary conditions \n")
    
    # # ensure these are jax arrays already
    # old_u = heq.u

    # heq.calculate_h()
    # u_bc = heq.apply_boundary_conditions(heq.u)
    # print("any NaN in h_boundary?", bool(np.any(np.isnan(heq.h_boundary_values))))
    # print("any NaN in u_bc?", bool(np.any(np.isnan(u_bc))))

    # old_u = heq.u
    # tau = heq.thermal_alpha * heq.dt / (heq.dx * heq.dy)

    # def probe(v):
    #     h_top = heq.h_top(heq.X, v)
    #     u_next = heq.interior_update_pure(
    #         old_u, h_top, heq.X, heq.Y, heq.heat_gen_a, heq.heat_gen_b, heq.heat_gen_c,
    #         tau, heq.k, heq.dx, heq.dy, heq.height, heq.ext_T, heq.heat_generation_function
    #     )
    #     return np.sum(u_next)

    # print("grad:", jax.grad(probe)(1.0))
    # probe_jit = jax.jit(probe)
    # print("jit val:", probe_jit(1.0))
    # print("jit grad:", jax.grad(probe_jit)(1.0))

    # v0 = 1.0
    # h = 1e-2  # or 1e-1 if needed
    # fd = (probe(v0+h) - probe(v0-h)) / (2*h)
    # ad = jax.grad(probe)(v0)
    # print("AD:", float(ad), "FD:", float(fd), "diff:", float(ad-fd))

    print("Testing h_boundary \n")
    # optional debug
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)

    old_u = heq.u
    tau = heq.thermal_alpha * heq.dt / (heq.dx * heq.dy)

    h_bnd = heq.h_boundary(old_u)
    h_top = heq.h_top(heq.X, 1.0)  # fix v for this test

    def probe(a):
        e_dot = heq.heat_generation_function(heq.X, heq.Y, a, 0.0, 0.0)
        u_bc = HeatEquation2D.apply_boundary_conditions_pure(
            old_u, e_dot, h_bnd, h_top,
            tau, heq.k, heq.dx, heq.dy, heq.height, heq.ext_T
        )
        return np.sum(u_bc)

    print("grad:", jax.grad(probe)(1.0))
    probe_jit = jax.jit(probe)
    print("jit val:", probe_jit(1.0))
    print("jit grad:", jax.grad(probe_jit)(1.0))

    # def probe(v):

    #     h_top = heq.h_top(heq.X, v)
    #     h_bnd = heq.h_boundary(old_u)

    #     # temporarily assign for your current method implementation
    #     heq.h_top_values = h_top
    #     heq.h_boundary_values = h_bnd

    #     u_bc = heq.apply_boundary_conditions(old_u)
    #     return np.sum(u_bc)

    # print("grad:", jax.grad(probe)(1.0))

    # print("JIT test \n")
    # probe_jit = jax.jit(probe)
    # print("jit value:", probe_jit(1.0))
    # print("jit grad:", jax.grad(probe_jit)(1.0))

    
    # # Optional: catch the exact op that generates NaNs/infs during backward pass
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_debug_infs", True)

    # old_u = heq.u
    # tau = heq.thermal_alpha * heq.dt / (heq.dx * heq.dy)

    # # e_dot should be JAX-friendly (your heat_generation_function must use jnp ops)
    # e_dot = heq.heat_generation_function(heq.X, heq.Y, heq.heat_gen_a, heq.heat_gen_b, heq.heat_gen_c)

    # # h_bnd depends on old_u only (if you’re testing v-derivative)
    # h_bnd = heq.h_boundary(old_u)

    # def probe(v):
    #     h_top = heq.h_top(heq.X, v)
    #     u_bc = heq.apply_boundary_conditions_pure(
    #         old_u, e_dot, h_bnd, h_top,
    #         tau, heq.k, heq.dx, heq.dy, heq.height, heq.ext_T
    #     )
    #     return np.sum(u_bc)   # scalar

    # # Eager grad
    # print("grad:", jax.grad(probe)(1.0))

    # # JIT + grad
    # probe_jit = jax.jit(probe)
    # print("jit value:", probe_jit(1.0))
    # print("jit grad:", jax.grad(probe_jit)(1.0))

    # v0 = 1.0
    # h = 1e-3
    # fd = (probe(v0+h) - probe(v0-h)) / (2*h)
    # ad = jax.grad(probe)(v0)
    # print("AD:", float(ad), "FD:", float(fd), "diff:", float(ad - fd))


    # def J_of_x(x):
    #     heq.reset()
    #     heq.set_fan_velocity(x[0])
    #     heq.set_heat_generation(heat_generation_function, x[1], x[2], x[3])
    #     heq.solve_until_steady_state()
    #     return w1*np.max(heq.u)/273 - w2*heq.fan_efficiency

    # ## outputs central difference partial derivative w.r.t. ith design variable
    # def FD_derivative(J, x, idx, h):
    #     x_plus = x.copy()
    #     x_minus = x.copy()

    #     x_plus[idx] += h
    #     x_minus[idx] -= h

    #     central_diff_derivative = (J(x_plus) - J(x_minus)) / (2 * h)
    #     return central_diff_derivative

    # ## Setting frame at optimal values 
    #         # [v_opt,           a_opt,                   b_opt,              c_opt] from running optimziation in part (b)
    # x_opt = [20.00074766692191, -58.84393563772187, -58.89571954019555, 153324.33094301834]
    # print(f"v: {x_opt[0]} m/s, ", f"a: {x_opt[1]}, ", f"b: {x_opt[2]}, ", f"c: {x_opt[3]}", f"\n")
    # heq.verbose = False

    # J_opt = J_of_x(x_opt)
    # print("J at optimal point (start) = ", J_opt)

    # # idx = 0 -> v
    # # idx = 1 -> a
    # # idx = 2 -> b
    # # idx = 3 -> c

    # for idx in [0, 1, 2, 3]:
    #     cur_list = []
    #     for h in eps_vals:
    #         point_deriv = FD_derivative(J_of_x, x_opt, idx, h)
    #         print("dJ/d_ = ", point_deriv)
    #         cur_list.append(point_deriv)
    #         print("ran for h=", h)
    #     FD_grad_J.append(cur_list)
    
    # print("dJ/dv: ", FD_grad_J[0], "\n")
    # print("dJ/da: ", FD_grad_J[1], "\n")
    # print("dJ/db: ", FD_grad_J[2], "\n")
    # print("dJ/dc: ", FD_grad_J[3], "\n")


    # #======== AD w/ Jax ========
    # idx = 3   #Using design variable a
    # # h_vals = []
    # # J_vals = []

    # def derivative_surrogate(x_opt, idx):

    #     h_vals = []
    #     J_vals = []

    #     for h in eps_vals:
    #         x_plus = x_opt.copy()
    #         x_minus = x_opt.copy()

    #         x_plus[idx] += h
    #         x_minus[idx] -= h

    #         h_vals.append(+h)
    #         J_vals.append(J_of_x(x_plus))

    #         h_vals.append(-h)
    #         J_vals.append(J_of_x(x_minus))

    #     h_vals = np.array(h_vals)
    #     J_vals = np.array(J_vals)

    #     # Approximating the gradient using quadratic
    #     # Design matrix
    #     A = np.column_stack([
    #         np.ones_like(h_vals),
    #         h_vals,
    #         h_vals**2
    #     ])

    #     # Least squares fit
    #     beta, residuals, rank, svals = np.linalg.lstsq(A, J_vals, rcond=None)

    #     beta0, beta1, beta2 = beta

    #     return beta 

    # def surrogate_J(h, beta):
    #     return beta[0] + beta[1]*h + beta[2]*h**2

    # dJdh_AD = jax.grad(surrogate_J)(0.0, jnp.array(derivative_surrogate(x_opt, idx)))
    # print("AD-based derivative:", dJdh_AD)

    # plt.figure(figsize=(12,8))
    # fig, ax = plt.subplots(4,1)
    # ax[0].plot(eps_vals, FD_grad_J[0], label="dJ/dv")
    # ax[1].plot(eps_vals, FD_grad_J[1], label="dJ/da")
    # ax[2].plot(eps_vals, FD_grad_J[2], label="dJ/db")
    # ax[3].plot(eps_vals, FD_grad_J[3], label="dJ/dc")
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')
    # ax[2].set_xscale('log')
    # ax[3].set_xscale('log')
    # ax[3].set_xlabel("step size, h")
    # fig.supylabel("Partial derivatives, $\partial J/ \partial r$")
    # plt.title("Stability of Central Finite-Difference Approximations")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # x_array = np.array(x_vals)
    # x_array = x_array.astype(float)

    # plt.figure(figsize=(12,8))
    # plt.plot(iterations, x_array[:,0], label="v (fan)")
    # plt.plot(iterations, x_array[:,1], label="a")
    # print("a:", x_array[:,1] )
    # print("b:", x_array[:,2] )
    # plt.plot(iterations, x_array[:,2], label="b")
    # plt.plot(iterations, x_array[:,3], label="c")
    # plt.yscale('symlog')
    # plt.legend()
    # plt.title("Parameter Convergence")
    # plt.grid(True)
    # plt.show()


    # grad_iterations = np.arange(len(gradL_vals))
    # plt.figure(figsize=(12,8))
    # print("gradL_vals:", gradL_vals)
    # plt.plot(grad_iterations, gradL_vals)
    # plt.xlabel("Iteration")
    # plt.ylabel("||grad L||")
    # plt.yscale('log')
    # #plt.legend()
    # plt.title("Lagrangian Convergence")
    # plt.grid(True)
    # plt.show()