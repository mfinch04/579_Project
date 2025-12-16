# Imports
import numpy as np # numpy for vectorization
from collections.abc import Callable # For type hints
import matplotlib.pyplot as plt
from scipy import optimize
import jax
import jax.numpy as jnp
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
        self.heat_generation_function = heat_generation_function
        self.heat_gen_a = a
        self.heat_gen_b = b
        self.heat_gen_c = c
        heat_generation_matrix = self.heat_generation_function(self.X,self.Y,self.heat_gen_a,self.heat_gen_b,self.heat_gen_c) * self.dx * self.dy *self.height
        i0, iN, j0 ,jN = 0, self.n_x - 1, 0 , self.n_y - 1
        # Boundaries with one side
        heat_generation_matrix[i0,:] /= 2
        heat_generation_matrix[iN,:] /= 2
        heat_generation_matrix[j0,:] /= 2
        heat_generation_matrix[jN,:] /= 2
        # Boundaries with two sides
        heat_generation_matrix[i0,j0] /= 2
        heat_generation_matrix[iN,jN] /= 2
        heat_generation_matrix[iN,j0] /= 2
        heat_generation_matrix[i0,jN] /= 2
        self.heat_generation_total = np.sum(np.sum(heat_generation_matrix))

    def set_fan_velocity(self, v: float):
        """Sets the fan velocity

        Parameters

        ------

        v (float): Variable associated with the fan velocity
        """
        self.v = v
        self.fan_efficiency = self.fan_efficiency_func(self.v)


    def h_boundary(self,u: np.ndarray):
        """Calculates the convective heat transfer coefficient at the boundaries

        Parameters

        ------

        u (np.ndarray): Current Temperature Mesh
        """
        beta = 1/((u+self.ext_T)/2)
        rayleigh = 9.81*beta*(u-self.ext_T)*self.dx**3/(self.ext_nu**2)*self.ext_Pr
        nusselt = (0.825 + (0.387*rayleigh**(1/6))/
                   (1+(0.492/self.ext_Pr)**(9/16))**(8/27))**2
        return nusselt*self.ext_k/self.dx

    def h_top(self,x: np.ndarray,u):
        """Calculates the convective heat transfer coefficient from the fan velocity

        Parameters

        ------

        x (np.ndarray): x position

        u (np.ndarray): UNUSED
        """
        Rex = self.v*x/self.ext_nu
        r,c = Rex.shape
        Nux = np.zeros((r,c))
        for i in range(r):
            for j in range(c):
                if Rex[i,j] < 5E5:
                    Nux[i,j] = 0.332*Rex[i,j]**0.5*self.ext_Pr**(1/3)
                else:
                    Nux[i,j] = 0.0296*Rex[i,j]**0.8*self.ext_Pr**(1/3)
        h = Nux*self.ext_k/(x + 1E-5)
        return h

    def calculate_h(self):
        """Calculates all necessary convective heat transfer coefficients"""
        self.h_top_values = self.h_top(self.X,self.u)
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
        # Left
        self.u[i0,1:-1] = (old_u[i0,1:-1] +
                            2 * tau * self.h_boundary_values[i0,1:-1]/self.k * self.dy * (self.ext_T - old_u[i0,1:-1]) +
                            tau * self.dx * (old_u[i0,2:] - old_u[i0,1:-1]) / self.dy +
                            tau * self.dx * (old_u[i0,1:-1] - old_u[i0,2:]) / self.dy +
                            2 * tau * self.dy * (old_u[i0 + 1, 1:-1] - old_u[i0, 1:-1]) / self.dx +
                            tau * self.h_top_values[i0,1:-1]/self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[i0,1:-1]) +
                            tau * e_dot[i0,1:-1] / self.k * self.dx * self.dy)

        # Right
        self.u[iN, 1:-1] = (old_u[iN, 1:-1] +
                            2 * tau * self.h_boundary_values[iN, 1:-1] / self.k * self.dy * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * self.dx * (old_u[iN, 2:] - old_u[iN, 1:-1]) / self.dy +
                            tau * self.dx * (old_u[iN, 1:-1] - old_u[iN, 2:]) / self.dy +
                            2 * tau * self.dy * (old_u[iN- 1, 1:-1] - old_u[iN,1:-1]) / self.dx +
                            tau * self.h_top_values[iN, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * e_dot[iN, 1:-1] / self.k * self.dx * self.dy)

        # Bottom
        self.u[1:-1,j0] = (old_u[1:-1,j0] +
                            2 * tau * self.h_boundary_values[1:-1,j0] / self.k * self.dx * (self.ext_T - old_u[1:-1,j0]) +
                            tau * self.dy * (old_u[2:,j0] - old_u[1:-1,j0]) / self.dx +
                            tau * self.dy * (old_u[1:-1,j0] - old_u[2:,j0]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1,j0 + 1] - old_u[1:-1,j0]) / self.dy +
                            tau * self.h_top_values[1:-1,j0] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[1:-1,j0]) +
                            tau * e_dot[1:-1,j0] / self.k * self.dx * self.dy)

        # Top
        self.u[1:-1,jN] = (old_u[1:-1,jN] +
                            2 * tau * self.h_boundary_values[1:-1,jN] / self.k * self.dx * (self.ext_T - old_u[1:-1,jN]) +
                            tau * self.dy * (old_u[2:,jN] - old_u[1:-1,jN]) / self.dx +
                            tau * self.dy * (old_u[1:-1,jN] - old_u[2:,jN]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1,jN - 1] - old_u[1:-1,jN]) / self.dy +
                            tau * self.h_top_values[1:-1,jN] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[1:-1,jN]) +
                            tau * e_dot[1:-1, jN] / self.k * self.dx * self.dy)

        ## Bottom Left Corner
        self.u[i0,j0] = (old_u[i0,j0] +
                         2 * tau * self.h_boundary_values[i0,j0] * self.dy / self.k * (self.ext_T - old_u[i0,j0]) +
                         2 * tau * self.h_boundary_values[i0,j0] * self.dx / self.k * (self.ext_T - old_u[i0,j0]) +
                         2 * tau * self.dx * (old_u[i0,j0+1] - old_u[i0,j0]) / self.dy +
                         2 * tau * self.dy * (old_u[i0+1,j0] - old_u[i0,j0]) / self.dx +
                         tau * self.h_top_values[i0,j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[i0,j0]) +
                         tau * e_dot[i0,j0] / self.k * self.dx * self.dy)
        ## Bottom Right Corner
        self.u[iN,j0] = (old_u[iN,j0] +
                         2 * tau * self.h_boundary_values[iN,j0] * self.dy / self.k * (self.ext_T - old_u[iN,j0]) +
                         2 * tau * self.h_boundary_values[iN,j0] * self.dx / self.k * (self.ext_T - old_u[iN,j0]) +
                         2 * tau * self.dx * (old_u[iN,j0+1] - old_u[iN,j0]) / self.dy +
                         2 * tau * self.dy * (old_u[iN-1,j0] - old_u[iN,j0]) / self.dx +
                         tau * self.h_top_values[iN,j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN,j0]) +
                         tau * e_dot[iN,j0] / self.k * self.dx * self.dy)
        ## Top Left Corner
        self.u[i0,jN] = (old_u[i0,jN] +
                         2 * tau * self.h_boundary_values[i0,jN] * self.dy / self.k * (self.ext_T - old_u[i0,jN]) +
                         2 * tau * self.h_boundary_values[i0,jN] * self.dx / self.k * (self.ext_T - old_u[i0,jN]) +
                         2 * tau * self.dx * (old_u[i0,jN-1] - old_u[i0,jN]) / self.dy +
                         2 * tau * self.dy * (old_u[i0+1,jN] - old_u[i0,jN]) / self.dx +
                         tau * self.h_top_values[i0,jN] / self.k * self.dx * self.dy / self.height  * (self.ext_T - old_u[i0,jN]) +
                         tau * e_dot[i0,jN] / self.k * self.dx * self.dy)
        ## Top Right Corner
        self.u[iN,jN] = (old_u[iN,jN] +
                         2 * tau * self.h_boundary_values[iN,jN] * self.dy / self.k * (self.ext_T - old_u[iN,jN]) +
                         2 * tau * self.h_boundary_values[iN,jN] * self.dx / self.k * (self.ext_T - old_u[iN,jN]) +
                         2 * tau * self.dx * (old_u[iN,jN-1] - old_u[iN,jN]) / self.dy +
                         2 * tau * self.dy * (old_u[iN-1,jN] - old_u[iN,jN]) / self.dx +
                         tau * self.h_top_values[iN,jN] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN,jN]) +
                         tau * e_dot[iN,jN] / self.k * self.dx * self.dy)
        return

    def step_forward_in_time(self):
        """Steps forward in time 1 timestep"""
        self.calculate_h()
        old_u = self.u.copy()
        self.apply_boundary_conditions(old_u)
        tau = self.thermal_alpha * self.dt / (self.dx * self.dy)
        self.u[1:-1, 1:-1] = (old_u[1:-1, 1:-1] +
                                    tau * (
                                            self.dy * (old_u[2:, 1:-1] - 2 * old_u[1:-1, 1:-1] + old_u[0:-2, 1:-1]) / self.dx  +
                                            self.dx * (old_u[1:-1, 2:] - 2 * old_u[1:-1, 1:-1] + old_u[1:-1, 0:-2]) / self.dy
                                    ) + tau * (self.h_top_values[1:-1, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[1:-1, 1:-1]) +
                                    self.dx * self.dy / self.k * self.heat_generation_function(self.X[1:-1, 1:-1],self.Y[1:-1, 1:-1],self.heat_gen_a,self.heat_gen_b,self.heat_gen_c)))
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
    FD_grad_J = [[], [], [], []]


    def J_of_x(x):
        heq.reset()
        heq.set_fan_velocity(x[0])
        heq.set_heat_generation(heat_generation_function, x[1], x[2], x[3])
        heq.solve_until_steady_state()
        return w1*np.max(heq.u)/273 - w2*heq.fan_efficiency

    ## outputs central difference partial derivative w.r.t. ith design variable
    def FD_derivative(J, x, idx, h):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[idx] += h
        x_minus[idx] -= h

        central_diff_derivative = (J(x_plus) - J(x_minus)) / (2 * h)
        return central_diff_derivative

    ## Setting frame at optimal values 
            # [v_opt,           a_opt,                   b_opt,              c_opt] from running optimziation in part (b)
    x_opt = [20.00074766692191, -58.84393563772187, -58.89571954019555, 153324.33094301834]
    print(f"v: {x_opt[0]} m/s, ", f"a: {x_opt[1]}, ", f"b: {x_opt[2]}, ", f"c: {x_opt[3]}", f"\n")
    heq.verbose = False

    # idx = 0 -> v
    # idx = 1 -> a
    # idx = 2 -> b
    # idx = 3 -> c
    # for idx in [0, 1, 2, 3]:
    #     for h in eps_vals:
    J_opt = J_of_x(x_opt)
    print("J at optimal point (start) = ", J_opt)

    dJ_da = FD_derivative(J_of_x, x_opt, 1, h=1e-2)
    print("h=0.01 dJ/da = ", dJ_da)
    dJ_da = FD_derivative(J_of_x, x_opt, 1, h=1e-3)
    print("h=0.001 dJ/da = ", dJ_da)
    dJ_da = FD_derivative(J_of_x, x_opt, 1, h=1e-4)
    print("h=0.0001 dJ/da = ", dJ_da)


    # plt.figure(figsize=(12,8))
    # plt.plot(iterations, obj_list, label="Objective")
    # print(type(obj_vals[0]))
    # print(type(obj_list[0]))
    # print("obj:", obj_list)
    # plt.plot(iterations, T_vals, label="Max Temperature")
    # plt.plot(iterations, eta_vals, label="Fan Efficiency")
    # plt.plot(iterations, power_vals, label="Constraint Value")
    # plt.yscale('symlog')
    # plt.legend()
    # plt.xlabel("Iteration")
    # plt.ylabel("Value")
    # plt.title("Convergence History")
    # plt.grid(True)
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