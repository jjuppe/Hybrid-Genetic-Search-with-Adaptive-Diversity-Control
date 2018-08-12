import numpy as np


def get_rank_array(arr, type="asc"):
    """
    This helper function returns an array indicating the rank of the value at each position

    Args:
        arr: NumpyArray
        type: asc or desc
    """

    length_array = len(arr)
    if type == "asc":
        sort_key = arr.argsort()
    else:
        sort_key = (-arr).argsort()

    rank_array = [True] * length_array
    for i in range(length_array):
        # get the position of the value in the original array that is at rank i
        val_pos = sort_key[i]
        # set the rank i at position val_pos
        rank_array[val_pos] = i

    return rank_array


def distance(start, stop):
    """
    Basic distance function for euclidean distance

    Args:
        start: VRPNode as start
        stop: VRPNode as stop

    Returns: euclidean distance

    """

    x_dist = np.subtract(start.x, stop.x)
    y_dist = np.subtract(start.y, stop.y)

    x_dist_square = np.square(x_dist)
    y_dist_square = np.square(y_dist)

    return np.sqrt(np.add(x_dist_square, y_dist_square))


def normalized_hamming_distance(instance1, instance2):
    """

    Annotation: The paper computes the difference based on the depot and service chromosome.
    As we simplified the problem to cut the time-window constraint and therefore only performed it on the depot chrom.

    Args:
        instance1: object of type instance
        instance2: object of type instance

    Returns:

    """

    depot_chromosome1 = instance1.depot_chromosome
    depot_chromosome2 = instance2.depot_chromosome
    n = instance1.vrp_data.nr_customers

    # get sum of equal depot_allocation of customers
    sum_val = 0
    for key in depot_chromosome1:
        if depot_chromosome1[key] != depot_chromosome2[key]:
            sum_val += 1

    return sum_val / n


def get_hemming_distance_matrix(merged_population):
    """
    Small helper function to get the normalized hemming distance

    Args:
        merged_population: merged population of instances of type "individual"

    Returns: distance matrix of all individuals contained

    """
    dist_matrix = []

    for indiv1 in merged_population:
        dist_arr = [normalized_hamming_distance(indiv1, indiv2) for indiv2 in merged_population]
        dist_matrix.append(dist_arr)
    return dist_matrix


def remove_column(matrix, i):
    """
    remove all column entries at position i
    """
    for row in matrix:
        try:
            del row[i]
        except IndexError:
            print(None)
            raise IndexError("blaaaa")


class Queue:
    """ Helper class for the Split algorithm

    This class implements a double-ended queue and some operations that are necessary
    for performing the Split algorithm

    """

    def __init__(self):
        self.queue = list()

    def pop_front(self):
        self.queue.__delitem__(0)

    def pop_back(self):
        self.queue.pop()

    def front(self):
        return self.queue[0]

    def front2(self):
        return self.queue[1]

    def back(self):
        return self.queue[-1]

    def push_back(self, t):
        self.queue.append(t)


if __name__ == "__main__":
    # TODO Test diversity population array
    print(None)
