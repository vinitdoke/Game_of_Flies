import numpy as np


def clusters_force_profile_distance(*args):
    pass


def all_force_functions(profile_name: str, n_types: int, *params):
    """
    return a matrix of functions 
    """
    n_types = len(params)
    for particle_type in params:
        for random in particle_type:
            print(random)

    if profile_name == "cluster_distance_input":
        pass
    elif profile_name == "cluster_position_input":
        pass

    return


if __name__ == "__main__":
    n_type = 2
    #     sample_input = np.random.uniform(0,1,(n_type, n_type, 4))
    sample_input = [[[1, 2, 3, 4], [5, 6, 7, 8]],
                    [[9, 10, 11, 12], [13, 14, 15, 16]]]
    all_force_functions("cluster_distance_input", 4, *sample_input)
    pass
