
import argh
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from concurrent.futures import ProcessPoolExecutor, as_completed


class Tracer:

    def __init__(self, U, x_coords, y_coords, dt):

        self.dt = dt
        self.vx = RectBivariateSpline(x_coords, y_coords, U[:,:,0])
        self.vy = RectBivariateSpline(x_coords, y_coords, U[:,:,1])

    def __call__(self, x, y, i, j):

        #xn, yn = rk4_trace(self.v, np.array([x,y]), self.dt, 10)

        f = lambda t,r: np.array([self.vx(r[0], r[1])[0][0], self.vy(r[0], r[1])[0][0]])
        xn, yn = solve_ivp(f, (0.0, self.dt), [x,y], t_eval=[self.dt]).y[:,0]

        return np.array([xn, yn, i, j])


def compute_phi(U, x_coords, y_coords, dt, nprocs):

    Nx, Ny, Nt, dummy = U.shape
    Phi = np.zeros(U.shape)

    for t in range(0, Nt):

        # initialize tracer
        tr = Tracer(U[:,:,t,:], x_coords, y_coords, dt)

        with ProcessPoolExecutor(max_workers=nprocs) as exe:

            # create futures set
            futures = set()

            # submit tracing jobs and assemble set of futures
            for i in range(0, Nx):
                for j in range(0, Ny):
                    x = x_coords[i]
                    y = y_coords[j]
                    futures.add(exe.submit(tr, x_coords[i], y_coords[j], i, j))

            # obtain results from futures as the traces are completed
            for fut in as_completed(futures):
                x, y, i, j = fut.result()
                Phi[int(i),int(j),t,:] = [x, y]


    return Phi


# this function gets called from CLI
def lagrange(filename, nprocs=1):

    """
    Reads velocity field from file and computes Phi at each point. Result is written back into file.
    """

    # read in U, x_coords, y_coords, and dt from file filename.h5
    U = np.zeros((20, 20, 20, 2))
    x_coords = np.linspace(0, 1, 20)
    y_coords = np.linspace(0, 1, 20)
    dt = 0.01

    # compute phi
    phi = compute_phi(U, x_coords, y_coords, dt, nprocs)

    # write phi as a new array into filename.h5





parser = argh.ArghParser()
parser.add_commands([lagrange])
if __name__ == '__main__':

    parser.dispatch()








#
# import matplotlib.pyplot as plt
# import math
#
# Nx = 20
# Ny = 20
# Nt = 2
# dt = 0.1
#
# U = np.zeros((Nx, Ny, Nt, 2))
# x_coords = np.linspace(0, 1, Nx)
# y_coords = np.linspace(0, 1, Ny)
#
# for i in range(0, Nx):
#     for j in range(0, Ny):
#         x = x_coords[i] - 0.5
#         y = y_coords[j] - 0.5
#         t = math.atan2(y, x)
#         r = np.sqrt(x**2 + y**2)
#         U[i,j,0,:] = r*np.array([-np.sin(t), np.cos(t)])
#         for t in range(1, Nt):
#             U[i,j,t,:] = U[i,j,0,:]
#
# phi = compute_phi(U, x_coords, y_coords, dt, 1)
#
#
# for i in range(0, Nx):
#     for j in range(0, Ny):
#         x = x_coords[i]
#         y = y_coords[j]
#         dx, dy = phi[i,j,0,:] - np.array([x,y])
#         plt.arrow(x, y, dx, dy)
#
# #
# #
#
# X, Y = np.meshgrid(x_coords, y_coords)
# Vx = U[:,:,0,0]
# Vy = U[:,:,0,1]
# Vx = phi[:,:,0,0] - X
# Vy = phi[:,:,0,1] - Y
# plt.quiver(X, Y, Vx, Vy, scale=1, scale_units='xy')#, angles='xy')
