from force_profiles import general_force_function
from state_parameters import initialise
from integrator import integrate

def main():
	# Initialise the state parameters
	n_type_arr = [100,100,100,100]
	state_dict = initialise(n_type_arr)

	n_iterations = 10000

