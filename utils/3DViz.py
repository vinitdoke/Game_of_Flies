import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

color_list = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple']
def setup_3Dplotter(n_types: int, limits=(100, 100, 100), dark_mode=False):
	if n_types > len(color_list):
		raise ValueError("Too many types for current color list")

	plt.ion()
	fig = plt.figure(dpi=100)

	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlim(0, limits[0])
	ax.set_ylim(0, limits[1])
	ax.set_zlim(0, limits[2])

	# turn off ticks
	ax.tick_params(axis='both', which='both', length=0)
	# set aspect ratio
	ax.set_aspect('equal')

	# DARK MODE
	if dark_mode:
		fig.patch.set_facecolor('black')
		ax.set_facecolor('black')
		ax.spines['bottom'].set_color('white')
		ax.spines['top'].set_color('white')
		ax.spines['left'].set_color('white')
		ax.spines['right'].set_color('white')
		# ax.tick_params(axis='x', colors='white')
		# ax.tick_params(axis='y', colors='white')
		ax.yaxis.label.set_color('white')
		ax.xaxis.label.set_color('white')

	scatters = []
	for i in range(n_types):
		scatters.append(ax.scatter([], [], [], s=2, c=color_list[i]))
	return fig, scatters


