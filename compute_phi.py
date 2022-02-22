
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from multiprocessing import Pool
import numpy as np


# combine flow field data into a callable object for tracing.
class Tracer:

    def __init__(self, U, x_coords, y_coords, dt):

        self.dt = dt
        self.interp = RectBivariateSpline(x_coords, y_coords, U)

    def __call__(self, x, y):
        return solve_ivp(lambda t, r: self.interp(r[0], r[1])[0], (0.0, self.dt), [x, y], t_eval=[self.dt]).y[:,0]


if __name__ == '__main__':

    # placeholder assignments (these will be read in from file)
    dt = 0.01
    Nx = 20
    Ny = 20
    Nt = 1
    x_coords = np.linspace(0, 1, Nx)
    y_coords = np.linspace(0, 1, Ny)
    U = np.zeros((Nx, Ny, Nt))

    # initialize Phi
    Phi = np.zeros((Nx, Ny, Nt, 2))

    for t in range(0, Nt):

        # TODO: Parallelize the (i,j) iterations
        tr = Tracer(U[:,:,t], x_coords, y_coords, dt)
        for i in range(0, Nx):
            for j in range(0, Ny):
                Phi[i,j,t,:] = tr(x_coords[i], y_coords[j])


    # save Phi to file
