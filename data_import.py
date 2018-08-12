from node import Depot
from node import Customer
from tools import distance
import numpy as np
import os


def binary_decoding(x):
    hex_visit_comb = int(x)
    str_binary_list = list("{0:b}".format(hex_visit_comb))
    binary_list = [int(y) for y in str_binary_list]
    return binary_list


class Data:
    def __init__(self, dat_path):
        """
        Base constructor of the VRP data used in our solver
        NOTE:   All ids set for all nodes are unique and start at 0.
                The min(customer_id) is max(depot_id) +1

        Args:
            dat_path: Path to the data file in txt format

        Class variables:
            type:          Description of problem type to which the data belongs
                            0 (VRP)
                            1 (PVRP)
                            2 (MDVRP)
                            3 (SDVRP)
                            4 (VRPTW)
                            5 (PVRPTW)
                            6 (MDVRPTW)
                            7 (SDVRPTW)

            nr_customers:   Number of customer nodes
            customers:      Customer nodes from VRPNode class
            nr_depots:      Number of depots
            depots:         Depot nodes from VRPNode class
            nr_periods:       Number of periods
            bks:            Best known solution

        """
        self.bks = None
        self.dat_path = dat_path

        dat_list = []
        with open(dat_path, "r") as file:
            for row in file:
                # get the row data and perform simple preprocessing
                row = row[:-1]  # remove '\n'
                row_list = row.split(" ")  # split at " "
                # cast all values to double and remove empty entries caused by multiple sapces
                row_list = [float(x) for x in row_list if x is not ""]
                dat_list.append(row_list)


        # 1) PROBLEM DESCRIPTION
        prob_desc = dat_list.pop(0)
        self.type = int(prob_desc[0])
        self.nr_vehicles = int(prob_desc[1])
        self.nr_customers = int(prob_desc[2])
        self.nr_depots = int(prob_desc[3])
        self.nr_periods = 1 # TODO: Remove hardcoding if necessary

        # 2) DEPOT DESCRIPTION
        id_counter = 0 # set id counter for all current nodes

        depots = {}
        for i in range(self.nr_depots):
            max_route_duration, max_vehicle_load = dat_list.pop(0)

            # set max route duration to infinite to allow all computations
            if max_route_duration == 0:
                max_route_duration = np.inf

            depot_location_info = dat_list.pop(-1)

            tmp_depot = Depot(id_counter, depot_location_info[1], depot_location_info[2], int(prob_desc[1]),
                              max_route_duration, max_vehicle_load)

            depots[id_counter] = tmp_depot
            id_counter += 1
        self.depots = depots

        # 3) CUSTOMER DATA
        customers = {}
        for i in range(self.nr_customers):
            raw_customer_data = dat_list.pop(0)
            customer_data = [id_counter]

            customer_data += raw_customer_data[1:7]
            customer_data[0] = int(customer_data[0]) # set ID as integer

            # for all visit combinations decode them into a binary array
            # NOTE: ALL OUR DATA IS CURRENTLY ONE PERIOD -> USELESS
            list_visit_comb = []
            for x in raw_customer_data[7:]:
                binary_list = binary_decoding(x)
                list_visit_comb.append(binary_list)
            # As it is currently only one redundant representation only select the first
            # customer_data.append(list_visit_comb)
            # TODO: Make sure that the rep has euqal length all the time
            customer_data.append([list_visit_comb[0]])

            customers[id_counter] = Customer(*customer_data)  # create new customer object based on given list
            id_counter += 1
        self.customers = customers

        # 4) SET DISTANCE MATRIX OF ALL NODES (based on id)
        self.distance_matrix = None
        self.set_distance_matrix()

    def set_distance_matrix(self):
        """
        Calculate the distance matrix of between all depots and customers with current sorting.
        (Default: ids are increasing)
        """

        # 1) initialize variables for solution
        nodes = self.depots.copy()
        nodes.update(self.customers)
        m = self.nr_depots + self.nr_customers # get number dimensionality
        distance_matrix = np.zeros((m,m))

        # 2) calculate the distance matrix for all values
        for n_id1 in nodes:
            for n_id2 in nodes:
                distance_matrix[n_id1,n_id2] = distance(nodes[n_id1], nodes[n_id2])

        self.distance_matrix = distance_matrix

    def set_bks(self, bks_path):
        """
        Set the currently best known solution

        Args:
            path: path to the bks file

        """
        #bks_path = bks_path[:-7] + "bks/" + bks_path[-7:-6] + bks_path[-6:-3] + "res"
        with open(bks_path, "r") as file:
            for row in file:
                bks_string = row.rstrip()
                bks_value = float(bks_string)
                break # get only the first entry

        self.bks = bks_value


# create data instances
if __name__ == "__main__":
    data_files = []
    cwd = os.getcwd()
    for i in range(1, 23):
        path = cwd + "/data/p" + str(i).zfill(2) + ".txt"  # iterate through files
        data_files = data_files + [Data(path)]

    for i in range(1, 10):
        path = cwd + "/data/pr" + str(i).zfill(2) + ".txt"  # iterate through files
        data_files = data_files + [Data(path)]
