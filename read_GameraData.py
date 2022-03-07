
import h5py 
import numpy as np

scratch_path = '/glade/scratch/mlmoses/' 
run_prmDct = {scratch_path+'test/Hall8.gam.h5': {'dt':0.5, 'tFin':60.0}}

def extract_GameraData(filename): 
    """
    Top Levels of f: 
    * 'Step#<timestepNUM>': Subgroup containing data at timestep timestepNUN
        * 'D' : density (shape =n-1xm-1)
        * 'Vx' : Velocity X-Component (shape =n-1xm-1)
        * 'Vy' : Velocity Y-Component (shape =n-1xm-1)
    * 'X': nxm Array containing the n X values in the xy spatial grid (values only vary in the X[i, :])
    * 'Y': nxm Array containing the m Y values in the xy spatial grid (values only vary in the Y[:, i])
    * 'dV': Array of values

    Output: 
        Vx_arr : n-1xm-1xk array containinng the Vx data for each of the k time steps at the center of each of the n X values and m Y values.
        Vy_arr : n-1xm-1xk array containinng the Vy data for each of the k time steps at the center of each of the n X values and m Y values.
    """
    dt = run_prmDct[filename]['dt']
    time_arr = np.arange(0, run_prmDct[filename]['tFin']+dt, dt)
    nSteps = len(time_arr)

    f = h5py.File(filename, 'r')

    X = f['X'][()]
    Y = f['Y'][()]
#    dV = f['dV'][()]
    Vx_arr = np.zeros((X.shape[0]-1, Y.shape[1]-1, len(time_arr)))
    Vy_arr = np.zeros((X.shape[0]-1, Y.shape[1]-1, len(time_arr)))
    for stepNum in np.arange(0, len(time_arr)): 
        stepT = 'Step#%s' % (stepNum)
#        Vx = f[stepT+'/Vx'][()] # Velocity X-Component 
#        Vy = f[stepT+'/Vy'][()] # Velocity Y-Component 
        Vx_arr[:, :, stepNum] = f[stepT+'/Vx'][()] # Velocity X-Component 
        Vy_arr[:, :, stepNum] = f[stepT+'/Vy'][()] # Velocity Y-Component 
    return dt, time_arr, X, Y, Vx_arr, Vy_arr

def import_GameraData(filename):
    """
    Import Gamera Data into the compute_phi.py script.
    """
    dt, time_arr, X, Y, Vx_arr, Vy_arr = extract_GameraData(filename)
    U = np.zeros((X.shape[0]-1, Y.shape[1]-1, len(time_arr), 2))
    U[:, :, :, 0] = Vx_arr
    U[:, :, :, 1] = Vy_arr 
    x_coords = X[0, 0:-1]
    y_coords = Y[0:-1, 0]
    x_coords = x_coords + np.diff(X[0, :])/2.
    y_coords = y_coords + np.diff(Y[:, 0])/2.
#    return dt, x_coords, y_coords, time_arr, X, Y, U
    return dt, x_coords, y_coords, time_arr, U
