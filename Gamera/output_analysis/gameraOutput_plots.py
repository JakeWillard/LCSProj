
#### File to examine h5 output from GAMERA runs
import h5py 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
# import sys

scratch_path = '/glade/scratch/mlmoses/' 
home_path = '/glade/u/home/mlmoses/'
fig_path = home_path+'LCS_project/kaiju/output_analysis/'

testID = 'test'
file_path = scratch_path+testID+'/'

filename = file_path+'Hall8.gam.h5'
f = h5py.File(filename, 'r')
"""
Top Levels of f: 
* 'Step#<timestepNUM>': Subgroup containing data at timestep timestepNUN
    * 'D' : density (shape =n-1xm-1)
    * 'Vx' : Velocity X-Component (shape =n-1xm-1)
    * 'Vy' : Velocity Y-Component (shape =n-1xm-1)
* 'X': nxm Array containing the n X values in the xy spatial grid (values only vary in the X[i, :])
* 'Y': nxm Array containing the m Y values in the xy spatial grid (values only vary in the Y[:, i])
* 'dV': Array of values

"""
dd = f['Step#0/D'][()] # Density at time step 0 
X = f['X'][()]
Y = f['Y'][()]
dV = f['dV'][()]

stepT = 'Step#100'
for stepNum in [0, 10, 25, 50, 100, 120]: 
	stepT = 'Step#%s' % (stepNum)
	#prmPlt = 'D'
	#Vx = f[stepT+'/'+prmPlt][()]
	dd = f[stepT+'/D'][()] # Density at time step 0 
	Vx = f[stepT+'/Vx'][()] # Velocity X-Component 
	Vy = f[stepT+'/Vy'][()] # Velocity Y-Component 

	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	pcol = ax.pcolormesh(X, Y, dd)
	fig.colorbar(pcol, ax=ax)
	ax.set_xlabel('X') 
	ax.set_ylabel('Y') 
	ax.set_title('Density at '+stepT)
	fig.savefig(fig_path+testID+'_figures/DensityAtStp%s_ColorPlot.png'%(stepT.partition('#')[-1]))

	fig1 = plt.figure(2)
	ax1 = fig1.add_subplot(111)
	ax1.quiver(X[0:-1, 0:-1], Y[0:-1, 0:-1], Vx, Vy)
	ax1.set_xlabel('X') 
	ax1.set_ylabel('Y') 
	ax1.set_title('Velocity Field at '+stepT)
	fig1.savefig(fig_path+testID+'_figures/Velocity_Field_Step%s.png' %(stepT.partition('#')[-1]))
	
	plt.close('all'); del fig, fig1, ax, ax1
