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

        # self.heat_gen_a = 0.0
        self.heat_gen_a = -58.84393563772187 # a_opt

        # self.heat_gen_b = 0.0
        self.heat_gen_b = -58.89571954019555 # b_opt

        # self.heat_gen_c = 0.0
        self.heat_gen_c = 153324.33094301834 # c_opt

        self.heat_generation_total = 0.0

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
        self.steady_state_error_list = []
        self.max_iter = 5E4
        self.init_condition = init_condition
        self.apply_initial_conditions()

        ## External Variables of Air
        self.ext_k = 0.02772  # W/m/K Thermal Coeffcient
        self.ext_Pr = 0.7215  # Prantl Number
        self.ext_nu = 1.506 * 10 ** (-5)  # m^2/s Kinematic Viscosity
        self.ext_T = 273 + 20  # K Temperature

        ## Fan Variables
        # self.v = 10 # m/s Air Velocity
        self.v = 20.00074766692191 # v_opt
        self.fan_efficiency_func = lambda v: -0.002* np.pow(v, 2.0) + 0.08*v
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
    
    def set_fan_velocity(self, v: float):
        """Sets the fan velocity

        Parameters

        ------

        v (float): Variable associated with the fan velocity
        """
        self.v = v
        self.fan_efficiency = self.fan_efficiency_func(self.v)
    
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
        x_safe = np.maximum(x, 1e-6)             # ensures x_safe > 0 everywhere
        Rex = v * x_safe / self.ext_nu

        Pr_term = self.ext_Pr ** (1.0 / 3.0)

        Nu_lam = 0.332 * np.sqrt(Rex) * Pr_term   # sqrt is clearer than power(.,0.5)
        Nu_tur = 0.0296 * np.power(Rex, 0.8) * Pr_term

        Nux = np.where(Rex < 5e5, Nu_lam, Nu_tur)

        h = Nux * self.ext_k / (x_safe + 1e-5)    # use x_safe here too
        return h

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
    
    @staticmethod
    def step_forward_in_time_pure(u, v, a, b, c, p):
        """
        Pure 1-step update.
        u: (nx, ny) field
        v,a,b,c: design vars
        p: dict of constants + functions
        returns: (u_next, err_inf)
        """

        # coefficients
        h_top = p["h_top_fun"](p["X"], v)     # (nx, ny)
        h_bnd = p["h_bnd_fun"](u)             # (nx, ny)
        e_dot = p["heat_fun"](p["X"], p["Y"], a, b, c)

        # boundary update (pure, already working)
        u_bc = HeatEquation2D.apply_boundary_conditions_pure(
            u, e_dot, h_bnd, h_top,
            p["tau"], p["k"], p["dx"], p["dy"], p["height"], p["ext_T"]
        )

        # interior update (pure, already working)
        u_next = HeatEquation2D.interior_update_pure(
            u_bc, h_top, p["X"], p["Y"], a, b, c,
            p["tau"], p["k"], p["dx"], p["dy"], p["height"], p["ext_T"], p["heat_fun"]
        )

        err_inf = np.max(np.abs(u_next - u))
        return u_next, err_inf
    
    @staticmethod
    def solve_fixed_steps(u0, x, p, n_steps: int):
        """
        u0: initial field
        x: [v,a,b,c]
        """
        v, a, b, c = x

        def body(_, u):
            u_next, _ = HeatEquation2D.step_forward_in_time_pure(u, v, a, b, c, p)
            return u_next

        return jax.lax.fori_loop(0, n_steps, body, u0)

def make_params(heq: HeatEquation2D):
    return {
        "X": heq.X,
        "Y": heq.Y,
        "dx": heq.dx,
        "dy": heq.dy,
        "k": heq.k,
        "height": heq.height,
        "ext_T": heq.ext_T,
        "tau": heq.thermal_alpha * heq.dt / (heq.dx * heq.dy),
        "dt": heq.dt,

        # supply pure function handles
        "heat_fun": heq.heat_generation_function,
        "h_top_fun": lambda X, v: heq.h_top(X, v),
        "h_bnd_fun": lambda u: heq.h_boundary(u),
    }

if __name__ == "__main__":
    # Physical Dimensions
    cpu_x = 0.04  # m
    cpu_y = 0.04  # m
    cpu_z = 0.04  # m
    N = 25

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
    eps_vals = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4] # x-axis of differential plots


    def objective_function(x):
        # x = [v, a, b, c]


        heq.set_fan_velocity(x[0])
        heq.set_heat_generation(heat_generation_function, x[1], x[2], x[3])

        #Reset time and temperature field
        heq.reset()

        #Solve PDE
        heq.solve_until_steady_state(tol=global_tolerance)

        #Compute objective 
        Tmax = np.max(heq.u)
        fan_eff = heq.fan_efficiency

        obj =  w1 * (Tmax / 273) - w2 * fan_eff

        x_vals.append(x)
        T_vals.append(Tmax)
        eta_vals.append(fan_eff)
        obj_vals.append(obj)
        power_vals.append(heq.heat_generation_total)

        return obj

    def objective(x, heq, p, n_steps=1000):
        u_final = heq.solve_fixed_steps(heq.u, x, p, n_steps)
        Tmax = np.max(u_final)

        v = x[0]
        fan_eff = heq.fan_efficiency_func(v)  # make sure this uses jnp ops if inside jit
        w1 = 0.2
        w2 = 1 - w1
        return w1 * Tmax - w2 * fan_eff

    ## Bounds for inputs
    bounds = [
        (0, 30),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (0, np.inf),
    ]

    p = make_params(heq)

    obj = lambda x: objective(x, heq, p, n_steps=1)

    obj_jit = jax.jit(obj)
    grad_obj = jax.jit(jax.grad(obj))

    def scipy_obj(x_np):
        return float(obj_jit(np.array(x_np)))

    def scipy_grad(x_np):
        return onp.array(grad_obj(np.array(x_np)), dtype=float)

    def constraint_one(x):
        heq.set_heat_generation(heat_generation_function, x[1], x[2], x[3])
        return 10 - heq.heat_generation_total
    
    def callback_func(xk, state):

        if hasattr(state, 'lagrangian_grad'):
            grad_L = state.lagrangian_grad
            grad_L_norm = np.linalg.norm(grad_L)
            gradL_vals.append(grad_L_norm)
        else:
            # Handle cases where the attribute might not exist (e.g., other methods or versions)
            print(f"Warning: callback did not provide 'lagrangian_grad'")


    ## Setting the constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_one},
    ]
    ## Creating the initial guess
    v0 = 10
    x0_heat = 0
    x0 = [v0, x0_heat * 10 ** 5, x0_heat * 10 ** 5, (156250 - 0.02 * x0_heat * 10 ** 5 - 0.02 * x0_heat * 10 ** 5)]
    heq.verbose = False
    ## Optimize
    optimization_result = optimize.minimize(
        scipy_obj,
        x0,
        jac=scipy_grad,
        method='trust-constr',
        bounds=bounds,
        constraints=constraints, 
        callback=callback_func,
        options={'verbose': 3}
    )
    ## Build optimal solution
    heq.set_fan_velocity(optimization_result.x[0])
    heq.set_heat_generation(heat_generation_function, optimization_result.x[1], optimization_result.x[2],
                            optimization_result.x[3])
    print(
        f"Optimization result: {objective_function(optimization_result.x)}\n"
        f"v: {optimization_result.x[0]} m/s, "
        f"a: {optimization_result.x[1]}, "
        f"b: {optimization_result.x[2]}, "
        f"c: {optimization_result.x[3]}"
        f"\n"
        f"Constraints:\n"
        f"Total Heat Generation: {heq.heat_generation_total} Constraint: {constraint_one(optimization_result.x)}\n"
    )
    ## Plot optimal solution
    fig, ax = plt.subplots()
    contour3 = ax.contourf(heq.X, heq.Y, heq.u - 273)
    fig.colorbar(contour3, ax=ax)
    plt.show()


    iterations = np.arange(len(obj_vals))
    obj_list = [float(v) for v in obj_vals]

    plt.figure(figsize=(12,8))
    plt.plot(iterations, obj_list, label="Objective")
    print(type(obj_vals[0]))
    print(type(obj_list[0]))
    print("obj:", obj_list)
    plt.plot(iterations, T_vals, label="Max Temperature")
    plt.plot(iterations, eta_vals, label="Fan Efficiency")
    plt.plot(iterations, power_vals, label="Constraint Value")
    plt.yscale('symlog')
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Convergence History")
    plt.grid(True)
    plt.show()

    x_array = np.array(x_vals)
    x_array = x_array.astype(float)

    plt.figure(figsize=(12,8))
    plt.plot(iterations, x_array[:,0], label="v (fan)")
    plt.plot(iterations, x_array[:,1], label="a")
    print("a:", x_array[:,1] )
    print("b:", x_array[:,2] )
    plt.plot(iterations, x_array[:,2], label="b")
    plt.plot(iterations, x_array[:,3], label="c")
    plt.yscale('symlog')
    plt.legend()
    plt.title("Parameter Convergence")
    plt.grid(True)
    plt.show()


    grad_iterations = np.arange(len(gradL_vals))
    plt.figure(figsize=(12,8))
    print("gradL_vals:", gradL_vals)
    plt.plot(grad_iterations, gradL_vals)
    plt.xlabel("Iteration")
    plt.ylabel("||grad L||")
    plt.yscale('log')
    #plt.legend()
    plt.title("Lagrangian Convergence")
    plt.grid(True)
    plt.show()

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
    
    def objective(x, heq, p, n_steps=5000):
        u_final = heq.solve_fixed_steps(heq.u, x, p, n_steps)
        Tmax = np.max(u_final)

        v = x[0]
        fan_eff = heq.fan_efficiency_func(v)  # make sure this uses jnp ops if inside jit
        w1 = 0.2
        w2 = 1 - w1
        return w1 * Tmax - w2 * fan_eff
    
    print("Setting up problem: heq")
    ## Problem Set up
    heq = HeatEquation2D(cpu_x,cpu_y,cpu_z, N,N,
                       k=k_si,rho=rho_si,cp=c_si,
                       init_condition=initial_condition)
    
    print("Setting params")
    p = make_params(heq)
    print("params = ", p)
    x0 = np.array([heq.v, heq.heat_gen_a, heq.heat_gen_b, heq.heat_gen_c])
    x0 = np.asarray(x0, dtype=np.float32)
    x_opt = np.array([20.00074766692191, -58.84393563772187, -58.89571954019555, 153324.33094301834])
    x_opt = np.asarray(x_opt, dtype=np.float32)
    

    ## Setting objective function
    heq.max_iter = 5E5
    
    global_tolerance = 1E-3
    
    x_vals = []
    obj_vals = []
    grad_vals = []
    gradL_vals = []
    power_vals = []
    T_vals = []
    eta_vals = []
    eps_vals = np.geomspace(1e-5, 1e-1, 50)
    FD_grad_J = []


    obj = lambda x: objective(x, heq, p, n_steps=1)

    # print(heq.steady_state_error_list)

    obj_jit = jax.jit(obj)
    grad_obj = jax.jit(jax.grad(obj))

    print("val:", obj_jit(x_opt))
    print("grad:", grad_obj(x_opt))
    print("jax.grad:", jax.grad(obj)(x_opt))