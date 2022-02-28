
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from multiprocessing import Pool
import numpy as np
# import matplotlib.pyplot as plt


# combine flow field data into a callable object for tracing.
class Tracer:

    def __init__(self, U, x_coords, y_coords, dt):

        self.dt = dt
        self.interp_x = RectBivariateSpline(x_coords, y_coords, U[:,:,0])
        self.interp_y = RectBivariateSpline(x_coords, y_coords, U[:,:,1])

    def __call__(self, x, y):
        return solve_ivp(lambda t, r: [self.interp_x(r[0], r[1])[0][0], self.interp_y(r[0], r[1])[0][0]], (0.0, self.dt), [x, y], t_eval=[self.dt]).y[:,0]


if __name__ == '__main__':

    # placeholder assignments (these will be read in from file)
    dt = 0.01
    Nx = 20
    Ny = 20
    Nt = 1
    x_coords = np.linspace(0, 1, Nx)
    y_coords = np.linspace(0, 1, Ny)
    U = np.zeros((Nx, Ny, Nt, 2))

    # initialize Phi
    Phi = np.zeros((Nx, Ny, Nt, 2))

    for t in range(0, Nt):

        # TODO: Parallelize the (i,j) iterations
        tr = Tracer(U[:,:,t,:], x_coords, y_coords, dt)
        for i in range(0, Nx):
            for j in range(0, Ny):
                Phi[i,j,t,:] = tr(x_coords[i], y_coords[j])


    # save Phi to file
    
    # A bit of fake code to explain how to use the functions:
    
    # Let's say that we decided that a finite time integration of 7 timesteps is a good amount
    # Let one timestep be dT.
    # The forward integration code would look like this:
    
    Phi_7timesteps = np.zeros((Nx, Ny, Nt, 2))
    ftles = np.zeros((Nx, Ny, Nt))
    for i in range(Nt-7):
        Phi_7timesteps[:,:,i,:] = composite_phis(Phi[:,:,i:(i+7+1),:],x_coords,y_coords)
    for i in range(Nt-7):
        ftles[:,:,i] = compute_FTLES(Phi_7timesteps[:,:,i,:],x_coords,y_coords,dT)
    # We could combine the loops if we wanted ofc
    
    # The backwards integration code would look like
    
    Phi_reverse_7timesteps = np.zeros((Nx, Ny, Nt, 2))
    ftles = np.zeros((Nx, Ny, Nt))
    for i in range(Nt-7):
        Phi_reverse_7timesteps[:,:,i+7,:] = composite_phis(Phi_reverse[:,:,i:(i+7+1),:][:,:,::-1,:],x_coords,y_coords)
    for i in range(Nt-7):
        ftles[:,:,i] = compute_FTLES(Phi_reverse_7timesteps[:,:,i,:],x_coords,y_coords,-dT)




































# Computes the finite time Lyapunov exponents for a particular map phi (single timestep)
# Optionally plots as a sanity check

# It takes in phi, rangex, rangey, deltaT
# phi: the map (final positions of test particles) for a single start time, with its values arranged on the grid.
# It is assumed to be indexed as [initial x coord of test particle, initial y coord of test particle, map component]
# Where map component = 0 for x, and = 1 for y.
# rangex: the ordered set of x values of points on the grid
# rangey: the ordered set of y values of points on the grid
# deltaT: the integration time

# Returns a masked array of the FTLEs, arranged properly along the grid.
# Invalid points (so far, just sources or sinks and the boundary) are masked

def compute_FTLES(phi,rangex,rangey,deltaT):
    # Breaking up phi into components
    
    phix = phi[:,:,0]
    phiy = phi[:,:,1]

    
    # Generating grid from x and y values
    [ygrid,xgrid] = np.meshgrid(rangey,rangex)
    
    # Getting grid shape
    gridshape = xgrid.shape
    
    # Grid of zeros, with the correct shape
    zerogrid = np.zeros(gridshape)
    
    # Calculating finite difference jacobian components (but with zeros on the boundary, since it's not defined there)
    Jxx = np.copy(zerogrid)
    Jxy = np.copy(zerogrid)
    Jyx = np.copy(zerogrid)
    Jyy = np.copy(zerogrid)

    # final separation in x for initial x separation of one x grid spacing
    Jxx[1:-1,1:-1] = (phix[2:,1:-1]-phix[:-2,1:-1])/(xgrid[2:,1:-1]-xgrid[:-2,1:-1])
    # final separation in y for initial x separation of one x grid spacing
    Jyx[1:-1,1:-1] = (phiy[2:,1:-1]-phiy[:-2,1:-1])/(xgrid[2:,1:-1]-xgrid[:-2,1:-1])
    # final separation in y for initial y separation of one y grid spacing
    Jyy[1:-1,1:-1] = (phiy[1:-1,2:]-phiy[1:-1,:-2])/(ygrid[1:-1,2:]-ygrid[1:-1,:-2])
    # final separation in x for initial y separation of one y grid spacing
    Jxy[1:-1,1:-1] = (phix[1:-1,2:]-phix[1:-1,:-2])/(ygrid[1:-1,2:]-ygrid[1:-1,:-2])

    # We flatten all the jacobian components into 1d arrays, arrange them into the order we want them in, then move the axes to form an array of 2x2 matrices
    # We can later reshape back to the grid shape to recover the original indexing
    J = np.moveaxis(np.asarray([[Jxx.reshape(-1), Jxy.reshape(-1)],[Jyx.reshape(-1),Jyy.reshape(-1)]]),-1,0)
    
    # Taking singular values, but not the corresponding vectors, and separating into max and mins
    svdmax,svdmin = np.linalg.svd(J,compute_uv=False).T

    # If this condition is true, we have found a source or a sink. They are masked so they aren't mistakenly characterized as 
    # part of material lines
    # This has a side effect of masking the boundary where the jacobians are all zero, which is nice.
    badcondition = ((svdmax < 1) | (svdmin > 1))
    
    # The properly shaped masked array of maximum singular values, corresponding to the grid
    svdmat = np.ma.masked_where(badcondition, svdmax).reshape(gridshape)
    
    # The finite time lyapunov exponents
    ftle = np.log(svdmat)/np.abs(deltaT)
    
    # Optional: plotting
    # Comment or uncomment as desired
    
    # decx=int(gridshape[0]/10)
    # decy=int(gridshape[1]/10)
    # 
    # plt.pcolormesh(xgrid[1:-1,1:-1],ygrid[1:-1,1:-1],np.log((svdmat[1:-1,1:-1])),shading='auto')
    # plt.colorbar()
    # plt.quiver(xgrid[::decx,::decy],ygrid[::decx,::decy],(phix-xgrid)[::decx,::decy],(phiy-ygrid)[::decx,::decy],color='red')
    
    return ftle
    
# Composits a set of Phis in the order that they're given to it
# Takes in: philist, rangex, rangey, order
# The philist is a numpy array, indexed like [initial x coord of test particle, initial y coord of test particle, time, map component]
# If they are positive-time phis, they are already sorted in the right order. If they are negative time then the order must be reversed,
# for example by doing Phi[:,:,::-1,:]
# rangex: the ordered set of x values of points on the grid
# rangey: the prdered set of y values of points on the grid
# interporder: what order interpolation to use. Default is cubic, you can also use linear.

# Returns a new phi that is the composition of the given phis
def composite_phis(philist,rangex,rangey,interporder='cubic'):
    
    # Separating out components and changing indexes so we can iterate over time
    phixlist = np.moveaxis(philist[:,:,:,0],-1,0)
    phiylist = np.moveaxis(philist[:,:,:,1],-1,0)
    
    # Defining grid
    [ygrid,xgrid] = np.meshgrid(rangey,rangex)

    # The first phi is the identity map, aka the grid
    phixc = xgrid
    phiyc = ygrid
    
    # For plotting purposes, uncomment if you want to plot
    # Decimation factor for drawing field lines and arrows
    # decx = int(xgrid.shape[0]/20)
    # decy = int(xgrid.shape[1]/20)
    
    # Iteratively compositing the phis
    for i in range(phixlist.shape[0]):
        # Interpolating the ith phi
        if interporder=='cubic':
            phixinterp = RectBivariateSpline(rangex,rangey,phixlist[i],kx=3,ky=3)
            phiyinterp = RectBivariateSpline(rangex,rangey,phiylist[i],kx=3,ky=3)
        elif interporder=='linear':
            phixinterp = RectBivariateSpline(rangex,rangey,phixlist[i],kx=1,ky=1)
            phiyinterp = RectBivariateSpline(rangex,rangey,phiylist[i],kx=1,ky=1)
        else:
            raise Exception('only linear and cubic interpolation supported')
            
        # Compositing the ith phi with the previous composition of phi_0, ..., phi_{i-1}
        phixcnew = phixinterp(phixc,phiyc,grid=False)
        phiycnew = phiyinterp(phixc,phiyc,grid=False)

        # Optional: plotting. This traces out flow lines. It will significantly slow things down!
        # Comment or uncomment as desired
        # for j in range(len(xgrid[::decx,::decy].reshape(-1))):
        #     plt.plot([phixc[::decx,::decy].reshape(-1)[j],phixcnew[::decx,::decy].reshape(-1)[j]],[phiyc[::decx,::decy].reshape(-1)[j],phiycnew[::decx,::decy].reshape(-1)[j]],color='red')
        # if i == (phixlist.shape[0]-1):
        #     plt.quiver(xgrid[::decx,::decy],ygrid[::decx,::decy],(phix-xgrid)[::decx,::decy],(phiy-ygrid)[::decx,::decy])

        # Updating the composited phi to include the contribution of phi_i
        phixc = phixcnew
        phiyc = phiycnew
    # Creating the new phi
    
    phic = np.moveaxis(np.asarray([phixc,phiyc]),0,-1)
    
    # Checking to see what fraction of particles advected past the simulation boundary
    # This can be commented out with no issue
    ################################################################################################################################
    ################################################################################################################################
    boundaryx = [rangex[1],rangex[-2]]
    boundaryy = [rangey[1],rangey[-2]]
    phixtrim = phic[1:-1,1:-1,0]
    phiytrim = phic[1:-1,1:-1,1]
    nparticles = (len(rangex)-1)*(len(rangey)-1)
    
    oobx = ((phixtrim<boundaryx[0]) | (phixtrim>boundaryx[1]))
    ooby = ((phiytrim<boundaryy[0]) | (phiytrim>boundaryy[1]))
    oob = (oobx | ooby)

    nout = np.count_nonzero(oob)
    fracout = nout/nparticles
    
    print('fraction of particles lost:')
    print(fracout)
    ################################################################################################################################
    ################################################################################################################################

    
    return phic
