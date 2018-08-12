import random
import copy

import sys
import time
from functools import wraps
import logging
import collections

from tools import Queue


def timethis(func):
    """
    small helper function to time some methods

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


def random_instance(vrp_data, solver_instance):
    """
    Generate a random solution based on a given data object (might be infeasible)
    Args:
        vrp_data: data object in form of a MDPVP

    Returns: randomized chromosome

    """
    depots = vrp_data.depots
    customers = vrp_data.customers

    # 1) Create depot chromosome
    depot_chromosome = {}
    depot_index = list(depots.keys())
    for c_id in customers:
        depot_chromosome[c_id] = random.choice(depot_index)

    # 2) Create the grand tour chromosome
    gt_chromosome = {x: [] for x in depots}

    # create a randomly initialized gt_chromosome
    tmp = depot_chromosome.copy()

    customer_ids = list(customers.keys())
    random.shuffle(customer_ids)
    for c_id in customer_ids:
        # fill the depot grand tour chromosome in random order based on the previous random selection
        d_id = depot_chromosome[c_id]
        gt_chromosome[d_id] += [c_id]

    return Individual(vrp_data, solver_instance, depot_chromosome, gt_chromosome)


class Individual:
    def __init__(self, vrp_data, solver_instance, depot_chromosome, gt_chromosome, length=None):

        self.vrp_data = vrp_data
        self.solver_inst = solver_instance
        # CHROMOSOME
        self.depot_chromosome = depot_chromosome
        self.gt_chromosome = gt_chromosome  # a list of dictionaries
        self.solution = self.split()

        # CHROMOSOME DESCRIPTION
        self.length = length
        self.penalty_duration = 0
        self.penalty_load = 0
        self.sub_tour_lengths = None
        self.sub_tour_loads = None
        self.sub_tour_service_times = None
        self.sub_tour_omega_loads = None
        self.sub_tour_omega_durations = None
        self.feasibility = None
        self.evaluate_solution()

        self.fitness = None
        self.value = None

    # @timethis
    def evaluate_solution(self):
        """
        Set evaluation parameters length, penalty,
        Returns:

        """

        # small init of used variables
        solution = self.solution
        depots = self.vrp_data.depots

        length = 0
        penalty_duation = 0
        penalty_load = 0

        sub_tour_lengths = {d_id: [] for d_id in solution}
        sub_tour_loads = {d_id: [] for d_id in solution}
        sub_tour_service_times = {d_id: [] for d_id in solution}
        sub_tour_omega_loads = {d_id: [] for d_id in solution}
        sub_tour_omega_durations = {d_id: [] for d_id in solution}

        for d_id in solution:
            for sub_tour in solution[d_id]:
                sub_tour_length = self.get_sub_tour_length(d_id, sub_tour)
                sub_tour_load = self.get_sub_tour_load(sub_tour)
                sub_tour_service_time = self.get_sub_tour_service_time(sub_tour)

                # create penalty values
                omega_duration = max(0, (sub_tour_length + sub_tour_service_time) - depots[d_id].max_route_duration)
                omega_load = max(0, sub_tour_load - depots[d_id].max_vehicle_load)

                # save the values
                sub_tour_lengths[d_id].append(sub_tour_length)
                sub_tour_loads[d_id].append(sub_tour_load)
                sub_tour_service_times[d_id].append(sub_tour_service_time)
                sub_tour_omega_loads[d_id].append(omega_load)
                sub_tour_omega_durations[d_id].append(omega_duration)

                length += sub_tour_length
                penalty_duation += omega_duration
                penalty_load += omega_load

        # calculate penalties
        self.sub_tour_lengths = sub_tour_lengths
        self.sub_tour_loads = sub_tour_loads
        self.sub_tour_service_times = sub_tour_service_times
        self.sub_tour_omega_loads = sub_tour_omega_loads
        self.sub_tour_omega_durations = sub_tour_omega_durations
        self.length = length
        self.penalty_duration = penalty_duation
        self.penalty_load = penalty_load
        self.value = length  # NECESSARY?

        if penalty_load > 0 or penalty_duation > 0:
            self.feasibility = False
        else:
            self.feasibility = True

        return length + penalty_load + penalty_duation

    # TOUR HELPER FUNCTIONS
    def get_sub_tour_length(self, depot_id, tour):
        """
        Calculate the tour distance

        Args:
            distance_matrix: distance matrix of all nodes
            depot_id
            tour: Iterable with the depot on position 0

        Returns: tour distance of all nodes

        """

        distance_matrix = self.vrp_data.distance_matrix
        nr_stops = len(tour)

        tour_distance = 0
        for i in range(1, nr_stops):
            start = tour[i - 1]
            stop = tour[i]
            tour_distance += distance_matrix[start, stop]  # NODE ID starts at 1

        # add the distance from the depot end to depot
        if nr_stops > 0:
            tour_distance += distance_matrix[depot_id, tour[0]]
            tour_distance += distance_matrix[tour[-1], depot_id]

        return tour_distance

    def get_sub_tour_load(self, tour):
        """
        Calculate the cumulative demand of our tour

        Args:
            tour: list of node ids

        Returns: double for cumulative demand

        """

        customers = self.vrp_data.customers

        tour_load = 0
        for c_id in tour:
            tour_load += customers[c_id].demand
        return tour_load

    def get_sub_tour_service_time(self, tour):
        """
        Calculate the cumulative demand of our tour

        Args:
            tour: list of node ids

        Returns: double for cumulative demand

        """
        customers = self.vrp_data.customers

        tour_service_time = 0
        for c_id in tour:
            tour_service_time += customers[c_id].service_duration
        return tour_service_time

    def get_reduced_length(self, customer_id, depot_id, tour, position):
        """
        Get the length that the tour would be reduced by if we would remove the customer at position p

        """
        distance_matrix = self.vrp_data.distance_matrix
        tour_length = len(tour)

        # case 0: The customer is the only customer in the set
        if tour_length == 1:
            return distance_matrix[depot_id, customer_id] * 2
        # case 1: The customer is at position 0
        elif position == 0:
            start = depot_id
            new_dest = tour[1]
        # case 2: The customer is between 2 other customers
        elif position < tour_length - 1:
            start = tour[position - 1]
            new_dest = tour[position + 1]
        # case 3: The customer is at the end
        elif position == tour_length - 1:
            start = tour[-2]
            new_dest = depot_id
        else:
            raise ValueError("No valid position found")
        return distance_matrix[start, customer_id] + distance_matrix[customer_id, new_dest] - distance_matrix[
            start, new_dest]

    def get_add_length(self, customer_id, depot_id, tour, position):
        """
        Get the length that the tour would be added to if we would add the customer at position p.

        """
        distance_matrix = self.vrp_data.distance_matrix
        tour_length = len(tour)

        # case 0: The customer will be the only customer in the set
        if tour_length == 0:
            return distance_matrix[depot_id, customer_id] * 2
        # case 1: The customer is at position 0
        elif position == 0:
            start = depot_id
            prev_dest = tour[0]
        # case 2: The customer is between 2 other customers
        elif position < len(tour):
            start = tour[position - 1]
            prev_dest = tour[position]
        # case 3: The customer is at the end
        elif position == len(tour):
            start = tour[-1]
            prev_dest = depot_id
        else:
            raise ValueError("No valid position found")
        return distance_matrix[start, customer_id] + distance_matrix[customer_id, prev_dest] - distance_matrix[
            start, prev_dest]

    def adjust_individual_remove_customer(self, depot_id, vehicle_index, position):
        """
        Adjust all parameters of an instance after removing elements from an parameter

        """
        sub_tour = self.solution[depot_id][vehicle_index]
        customer_id = sub_tour[position]
        customer = self.vrp_data.customers[customer_id]

        # 2) Adjust parameters
        # 2.1) Adjust length parameter
        reduced_length = self.get_reduced_length(customer_id, depot_id, sub_tour, position)
        self.length -= reduced_length

        # 2.2) Adjust load specific parameter
        reduced_load = customer.demand
        self.sub_tour_loads[depot_id][vehicle_index] -= reduced_load

        # 2.2) Adjust other sub_tour specific parameter
        self.sub_tour_lengths[depot_id][vehicle_index] -= reduced_length
        self.sub_tour_service_times[depot_id][vehicle_index] -= customer.service_duration

        # 2.3) Adjust omega parameters
        old_omega_duration = self.sub_tour_omega_durations[depot_id][vehicle_index]
        new_omega_duration = max(0, (old_omega_duration - reduced_length - customer.service_duration))

        old_omega_load = self.sub_tour_omega_loads[depot_id][vehicle_index]
        new_omega_load = max(0, (old_omega_load - reduced_load))
        self.sub_tour_omega_loads[depot_id][vehicle_index] = new_omega_load
        self.sub_tour_omega_durations[depot_id][vehicle_index] = new_omega_duration
        self.penalty_duration -= (old_omega_duration - new_omega_duration)
        self.penalty_load -= (old_omega_load - new_omega_load)

        # 4) Recalculate the feasibility of the solution
        if self.penalty_duration > 0 or self.penalty_load > 0:
            self.feasibility = False
        else:
            self.feasibility = True

        # 3) Fianlly! Adjust the solution representation
        self.gt_chromosome[depot_id].remove(customer_id)
        del self.solution[depot_id][vehicle_index][position]

    def adjust_individual_add_customer(self, customer_id, depot_id, vehicle_index, position):
        """
        Adjust all parameters of an instance after removing elements from an parameter

        """
        # 0) Get the depot base stats
        depot_base_stats = self.vrp_data.depots[0]
        max_route_duration = depot_base_stats.max_route_duration
        max_vehicle_load = depot_base_stats.max_route_duration

        sub_tour = self.solution[depot_id][vehicle_index]
        customer = self.vrp_data.customers[customer_id]

        # 2) Adjust parameters
        # 2.1) Adjust length parameter
        add_length = self.get_add_length(customer_id, depot_id, sub_tour, position)
        self.length += add_length

        # 2.2) Adjust load specific parameter
        add_load = customer.demand
        self.sub_tour_loads[depot_id][vehicle_index] += add_load

        # 2.2) Adjust other sub_tour specific parameter
        self.sub_tour_lengths[depot_id][vehicle_index] += add_length
        self.sub_tour_service_times[depot_id][vehicle_index] += customer.service_duration

        # 2.3) Adjust omega parameters (we already adjusted the parameters here! No need to additionally add length
        new_omega_duration = max(0, (
                self.sub_tour_lengths[depot_id][vehicle_index] + self.sub_tour_service_times[depot_id][
            vehicle_index])
                                 - max_route_duration)

        new_omega_load = max(0, (self.sub_tour_loads[depot_id][vehicle_index]) - max_vehicle_load)

        self.penalty_duration += (new_omega_duration - self.sub_tour_omega_loads[depot_id][vehicle_index])
        self.penalty_load += (new_omega_load - self.sub_tour_omega_loads[depot_id][vehicle_index])
        self.sub_tour_omega_loads[depot_id][vehicle_index] = new_omega_load
        self.sub_tour_omega_durations[depot_id][vehicle_index] = new_omega_duration

        # check again for feasibility
        if self.penalty_duration > 0 or self.penalty_load > 0:
            self.feasibility = False
        else:
            self.feasibility = True

        # Finally, adjust the solution representation
        self.depot_chromosome[customer_id] = [depot_id]
        self.gt_chromosome[depot_id].append(customer_id)
        self.solution[depot_id][vehicle_index].insert(position, customer_id)

    # @timethis
    def pattern_improvement(self):
        """
        This is the main function for pattern improvement

        The procedure is as follows:
         1) Draw a random customer
         2) Check if any position swap might improve the solution
         3) If not continue at 1)
            If yes reset the pool of customers and start at 1

        Stop: If the pool of customers is empty

        """
        # guarantee the correct starting position!
        self.evaluate_solution()

        def get_current_info(customer_id):
            """
            Get the value that would be reduced if we remove the customer_id from the current solution
            """
            nonlocal self
            depot_id = self.depot_chromosome[customer_id][0]
            sub_tours = self.solution[depot_id]

            for vehicle_id, sub_tour in enumerate(sub_tours):
                if customer_id in sub_tour:
                    position_index = sub_tour.index(customer_id)
                    # 1) Get the current base stats of this subtour
                    omega_duration = self.sub_tour_omega_durations[depot_id][vehicle_id]
                    omega_load = self.sub_tour_omega_loads[depot_id][vehicle_id]

                    # 2) adapt the basestats if this would be missing
                    # 2.1) adapt the duration omega
                    reduced_length = self.get_reduced_length(customer_id, depot_id, sub_tour, position_index)
                    reduced_service_time = self.vrp_data.customers[customer_id].service_duration
                    new_omega_duration = max(0, (omega_duration - reduced_length - reduced_service_time))

                    # 2.2) adapt the load omega
                    reduced_load = self.vrp_data.customers[customer_id].demand
                    new_omega_load = max(0, omega_load - reduced_load)

                    reduced_value = reduced_length + (omega_load - new_omega_load) * self.solver_inst.w_penalty_load + (
                            omega_duration - new_omega_duration) * self.solver_inst.w_penalty_duration
                    return depot_id, vehicle_id, position_index, reduced_value

        def get_best_insertion_value(customer_id, depot_id, vehicle_id):
            """
            This helper function is used to calculate the best insertion position if non previous info exists

            """
            nonlocal self

            # The goal is to improve this solution now
            distance_matrix = self.vrp_data.distance_matrix
            current_solution = self.solution
            best_position_index = 0
            best_value = sys.maxsize

            # 0) Get the depot base stats
            depot_base_stats = self.vrp_data.depots[0]
            max_route_duration = depot_base_stats.max_route_duration
            max_vehicle_load = depot_base_stats.max_vehicle_load

            # 1.1) Get the current base stats of this subtour
            tour = current_solution[depot_id][vehicle_id]

            length = self.sub_tour_lengths[depot_id][vehicle_id]
            load = self.sub_tour_loads[depot_id][vehicle_id]
            service_time = self.sub_tour_service_times[depot_id][vehicle_id]
            omega_load = self.sub_tour_omega_loads[depot_id][vehicle_id]
            omega_duration = self.sub_tour_omega_durations[depot_id][vehicle_id]

            # 1.2) get the current base stats of this instance
            w_penalty_load = self.solver_inst.w_penalty_load
            w_penalty_duration = self.solver_inst.w_penalty_duration

            # precalculate what we can
            add_service_time = self.vrp_data.customers[customer_id].service_duration
            add_load = self.vrp_data.customers[customer_id].demand
            new_omega_load = max(0, (load + add_load) - max_vehicle_load)
            for position_index in range(len(tour) + 1):
                # 2) adapt the basestats if this would be missing
                # 2.1) adapt the duration omega
                add_length = self.get_add_length(customer_id, depot_id, tour, position_index)
                new_omega_duration = max(0,
                                         (length + add_length + service_time + add_service_time) - max_route_duration)

                # 3) Get the change in value! (We know that the new omegas have to be striktly bigger (because we only add one element)
                add_value = add_length + (new_omega_load - omega_load) * w_penalty_load + (
                        new_omega_duration - omega_duration) * w_penalty_duration

                if add_value < best_value:
                    best_value = add_value
                    best_position_index = position_index

            return depot_id, vehicle_id, best_position_index, best_value

        nr_customers = self.vrp_data.nr_customers
        nr_depots = self.vrp_data.nr_depots
        nr_vehicles = self.vrp_data.nr_vehicles

        # Initialize the insertion documentation
        customer_ids = list(self.depot_chromosome.keys())
        depot_ids = list(range(0, nr_depots))
        # the documentation contains all best insertion information necessary
        documentation = [[[None for n in range(nr_vehicles)] for j in range(nr_depots)] for i in range(nr_customers)]

        # random iteration over all customers
        unvisited_index = list(range(0, nr_customers))
        nr_unvisited_customers = nr_customers
        last_cust = None
        start_time = time.time()
        while nr_unvisited_customers > 0 and time.time() - start_time <= 1:
            nr_unvisited_customers -= 1
            # get the random customer id
            index = random.choice(unvisited_index)
            unvisited_index.remove(index)

            customer_id = customer_ids[index]
            # get the value that we would reduce if we remove this customer_id from the solution
            current_info = get_current_info(customer_id)
            best_insertion_info = current_info
            # best_depot_id, best_vehicle_index, best_insertion_index, best_value

            for depot_id in depot_ids:
                customer_doc = documentation[index][depot_id]
                # for each depot check the best position
                for vehicle_id, insertion_info in enumerate(customer_doc):
                    if customer_id not in self.solution[depot_id][vehicle_id]:
                        # if the best insertion value is None we have to recalculate it
                        # If the customer is already part of the subtour we ignore it
                        if insertion_info is None:
                            insertion_info = get_best_insertion_value(customer_id, depot_id, vehicle_id)
                            customer_doc[vehicle_id] = insertion_info

                        # save the information if we found a new best insertion value
                        # can also be improved if the current position changes!
                        if insertion_info[3] < best_insertion_info[3]:
                            best_insertion_info = insertion_info
                documentation[index][depot_id] = customer_doc  # renew documentation

            # check if a modification would improve the value, if thats the case reset!
            if best_insertion_info[3] < current_info[3]:
                # if we performed a new modification reset the list of customers
                unvisited_index = list(range(0, nr_customers))
                nr_unvisited_customers = nr_customers

                # adjust the individual by removing the customer
                self.adjust_individual_remove_customer(*current_info[:3])
                # adjust the individual by adding the customer
                self.adjust_individual_add_customer(customer_id, *best_insertion_info[:3])

                # change the spots that we have to recalculate
                for cust_id in range(nr_customers):
                    # All values for the subtours have to be recalculated
                    documentation[cust_id][current_info[0]][current_info[1]] = None
                    documentation[cust_id][best_insertion_info[0]][best_insertion_info[1]] = None

                    # The position where we removed it from keeps its imaginary value
                    documentation[index][current_info[0]][current_info[1]] = current_info
                    # The position where we move it to will keep the new value (no change here)
                    documentation[index][best_insertion_info[0]][best_insertion_info[1]] = best_insertion_info

        self.evaluate_solution()

    # @timethis
    def route_improvement(self, load_penality_factor, length_penalty_factor):
        ''' function that improves trips at a depot to find a better solution

        The route improvement function is applied to the offspring created in the crossover. Based on
        Vidal et al., 2012 section 4.5 the education consists of a route improvement (RI) and pattern
        improvement (PI) procedure. They are applied in RI, PI, RI sequence. The route improvement uses nine
        different local search moves. They include insertions, swaps, 2 opt intraroute and interroute swaps.
        The route improvement is done for each customer in conjunction with nodes from its immediate neighbourhood.

        Args:
            load_penality_factor: penalty factor for excess load
            length_penalty_factor: penalty factour for excess duration

        Returns:

        '''

        def find_route(cust):
            for subtour_index, subtours in enumerate(solution):
                if cust in subtours:
                    return cust, subtours, subtour_index, subtours.index(cust), depot

        def insertion_1(u, v):
            # check for depot nodes
            if all(n not in [0, 1, 2, 3] for n in [u, v]):
                solution[routes[0][2]].remove(u)
                solution[routes[index][2]].insert(routes[index][3] + 1, u)  # does add afer index?

        def insertion_2(u, x, v):
            # check for depot nodes
            if all(n not in [0, 1, 2, 3] for n in [u, x, v]) and routes[0][2] != routes[index][2]:
                solution[routes[0][2]].remove(u)
                solution[routes[0][2]].remove(x)
                solution[routes[index][2]].insert(routes[index][3] + 1, x)
                solution[routes[index][2]].insert(routes[index][3] + 1, u)

        def insertion_3(u, x, v):
            # check for depot nodes
            if all(n not in [0, 1, 2, 3] for n in [u, x, v]):
                solution[routes[0][2]].remove(u)
                solution[routes[0][2]].remove(x)
                solution[routes[index][2]].insert(routes[index][3] + 1, u)
                solution[routes[index][2]].insert(routes[index][3] + 1, x)

        def swap_1(u, v):
            if all(n not in [0, 1, 2, 3] for n in [u, v]):
                solution[routes[0][2]].remove(u)
                solution[routes[index][2]].remove(v)
                solution[routes[0][2]].insert(routes[0][3], v)
                solution[routes[index][2]].insert(routes[index][3], u)

        def swap_2(u, x, v):
            if all(n not in [0, 1, 2, 3] for n in [u, x, v]):
                solution[routes[0][2]].remove(u)
                solution[routes[0][2]].remove(x)
                solution[routes[index][2]].remove(v)
                solution[routes[index][2]].insert(routes[index][3], x)
                solution[routes[index][2]].insert(routes[index][3], u)
                solution[routes[0][2]].insert(routes[0][2], v)

        def swap_3(u, x, v, y):
            if all(x not in [0, 1, 2, 3] for x in [u, x, v, y]):
                solution[routes[0][2]].remove(u)
                solution[routes[0][2]].remove(x)
                solution[routes[index][2]].remove(v)
                solution[routes[index][2]].remove(y)
                solution[routes[index][2]].insert(routes[index][3], x)
                solution[routes[index][2]].insert(routes[index][3], u)
                solution[routes[0][2]].insert(routes[0][2], y)
                solution[routes[0][2]].insert(routes[0][2], v)

        def two_opt_intraroute(u, x, v, y):
            if routes[0][2] == routes[index][2]:
                solution[routes[0][2]].remove(x)
                solution[routes[index][2]].remove(v)
                solution[routes[0][2]].insert(routes[0][2] + 1, v)
                solution[routes[index][2]].insert(routes[index][2] + 1, x)

        def two_opt_interroute_1(u, x, v, y):
            if routes[0][2] != routes[index][2]:
                solution[routes[0][2]].remove(x)
                solution[routes[index][2]].remove(v)
                solution[routes[0][2]].insert(routes[0][2] + 1, v)
                solution[routes[index][2]].insert(routes[index][2], x)

        def two_opt_interroute_2(u, x, v, y):
            if routes[0][2] != routes[index][2]:
                solution[routes[0][2]].remove(x)
                solution[routes[index][2]].remove(y)
                solution[routes[0][2]].insert(routes[0][2] + 1, y)
                solution[routes[index][2]].insert(routes[index][2], x)

        def evaluate_depot(solution, d_id, load_penalty_factor, length_penalty_factor):
            nonlocal self
            depots = self.vrp_data.depots
            customers = self.vrp_data.customers
            length = 0
            load = 0
            penalty_load = 0
            penalty_duration = 0
            for subtour in solution:
                sub_tour_length = self.get_sub_tour_length(d_id, subtour)
                sub_tour_service_times = sum([customers[cust].service_duration for cust in subtour])
                sub_tour_load = sum([customers[cust].demand for cust in subtour])

                # create penalty values
                omega_duration = max(0, (sub_tour_length + sub_tour_service_times) - depots[d_id].max_route_duration)
                omega_load = max(0, sub_tour_load - depots[d_id].max_vehicle_load)

                # update parameters
                length += sub_tour_length
                penalty_duration += omega_duration
                penalty_load += omega_load
                load += sub_tour_load

            # calculate penalties
            # return length + load_penalty_factor * penalty_load + length_penalty_factor * penalty_duration
            return length + load_penalty_factor * penalty_load + length_penalty_factor * penalty_duration

        data = self.vrp_data.distance_matrix
        giant_tour_chromosome = self.gt_chromosome
        for depot, tour in giant_tour_chromosome.items():
            for vertex in tour:
                # random neighbourhood size
                neighbourhood_size = min(int(0.4 * len(tour)), self.vrp_data.nr_vehicles)
                # determine distance to other nodes in the same depot
                distances_sorted = sorted([(data[vertex][x], x) for x in tour if x != vertex], key=lambda x: x[0])
                # only takes the nodes that are in neighbourhood size
                if len(distances_sorted) < neighbourhood_size:
                    continue
                neighbourhood = [distances_sorted.pop() for _ in range(neighbourhood_size)]
                solution = copy.deepcopy(self.solution[depot])
                a = [j for sub in solution for j in sub]
                # check if gt chromosome contains all nodes at depot that are also contained in solution
                if not collections.Counter(tour) == collections.Counter(a):
                    raise ValueError('Error! Non-identical gt chromosome and solution' + str(tour) + '\n' + str(a))
                random.shuffle(neighbourhood)
                # finds the different routes of each node in the neighbourhood
                routes = [find_route(cust=vertex)] + [find_route(neighbourhood[x][1]) for x in
                                                      range(neighbourhood_size)]
                for index, neighbour in enumerate(neighbourhood, 1):
                    solution = copy.deepcopy(self.solution[depot])
                    routes = [find_route(cust=vertex)] + [find_route(neighbourhood[x][1]) for x in
                                                          range(neighbourhood_size)]
                    # sucessor of u
                    if routes[0][3] <= len(routes[0][1]) - 2:
                        successor = routes[0][1][routes[0][3] + 1]  # check if the node is the last in the tour
                    else:
                        continue
                    # neighbour node v
                    neighbour = neighbour[1]
                    # successor of v
                    if routes[index][3] <= len(routes[index][1]) - 2:
                        successor_of_neighbour = routes[index][1][routes[index][3] + 1]
                    else:
                        continue
                    # check if any of those are the same
                    if len([vertex, successor, neighbour, successor_of_neighbour]) != len(
                            {vertex, successor, neighbour, successor_of_neighbour}):
                        continue
                    current_cost = evaluate_depot(solution, depot, load_penality_factor, length_penalty_factor)
                    # perform the operations randomly
                    available_methods = [0, 1, 2, 3, 4, 5, 6, 7]
                    random.shuffle(available_methods)
                    for i in available_methods:
                        # choose which route improvement operator is selected
                        method = i
                        solution = copy.deepcopy(self.solution[depot])
                        if method == 0:
                            insertion_1(vertex, neighbour)
                        elif method == 1:
                            insertion_2(vertex, successor, neighbour)
                        elif method == 2:
                            insertion_3(vertex, successor, neighbour)
                        elif method == 3:
                            swap_1(vertex, neighbour)
                        elif method == 4:
                            swap_2(vertex, successor, neighbour)
                        elif method == 5:
                            swap_3(vertex, successor, neighbour, successor_of_neighbour)
                        elif method == 6:
                            two_opt_intraroute(vertex, successor, neighbour, successor_of_neighbour)
                        elif method == 7:
                            two_opt_interroute_1(vertex, successor, neighbour, successor_of_neighbour)
                        elif method == 8:
                            two_opt_interroute_2(vertex, successor, neighbour, successor_of_neighbour)
                        new_cost = evaluate_depot(solution, depot, load_penality_factor, length_penalty_factor)
                        if new_cost < current_cost:
                            # self.solution[depot] = solution
                            # after = self.evaluate_solution()
                            self.solution[depot] = solution
                            break

    def split(self):
        """ linear split algorithm that finds optimal tour delimeters

        The split algorithm return an optimal solution for a given sequence of customers.
        This implementation is based on Vidal, 2015 which introduces a split algorithm
        solveable in linear time. It takes into account capacity constraints and limited
        vehicle fleet at the depots. The split algorithm is applied in the crossover
        as explained in Vidal et al., 2012 in section 4.4.


        Returns: a dictionary with the optimal trips at each depot.

        """

        def create_split_data(solution_instance):
            """ function that creates a data object necessary to perform the split algorithm

            This function creates a data object for the split algorithm

            data:
                [0] customer id
                [1] counter to perform split (requires ascending node ids)
                [2] distance to predecessors
                [3] distance to depot
                [4] demand of that customer
                [5] service duration at the customer

            Args:
                solution_instance: requires a giant tour chromosome

            Returns: data object that contains necessary information about the problem

            """
            # nr_depots = solution_instance.vrp_data.nr_depots

            distance_matrix = solution_instance.vrp_data.distance_matrix
            nodes = [[0] + solution_instance.gt_chromosome[x] for x in depots]

            data_complete = list()
            for index, i in enumerate(depots):
                data = [[0 for _ in range(6)] for _ in range(len(nodes[index]))]
                for j in range(1, len(nodes[index])):
                    data[j][0] = nodes[index][j]
                    data[j][1] = j
                    data[j][2] = distance_matrix[nodes[index][j - 1]][nodes[index][j]]
                    data[j][3] = distance_matrix[index][nodes[index][j]]
                    data[j][4] = solution_instance.vrp_data.customers[nodes[index][j]].demand
                    data[j][5] = solution_instance.vrp_data.customers[nodes[index][j]].service_duration
                data_complete.append(data)

            return data_complete

        def linear_split(data, Q_max, max_tour_duration, nr_vehicles):
            """
            # TODO Create docstring

            Args:
                child:

            Returns:

            """
            nonlocal self
            length_penalty = self.solver_inst.w_penalty_duration
            load_penalty = self.solver_inst.w_penalty_load

            def cost_achieved(data, i, x, Q, D, T):
                if Q[x] - Q[i] <= 2 * Q_max:
                    return cost(data, i, x, D, Q, T)
                return sys.maxsize

            def cost(data, i, j, D, Q, T):
                return data[i + 1][3] + D[j] - D[i + 1] + data[j][3] + T[j] - T[i] + load_penalty * max(
                    Q[j] - Q[i] - Q_max,
                    0) + length_penalty * max(
                    + T[j] + D[j] - D[i + 1] - T[i] - max_tour_duration, 0)

            def dominates(data, i, j, k, p, D, Q, T):
                if i <= j:
                    return p[k][i] + data[i + 1][3] - D[i + 1] + T[j] - T[i] + load_penalty * (Q[j] - Q[i]) <= p[k][j] + \
                           data[j + 1][3] - D[
                               j + 1]
                else:
                    return p[k][i] + data[i + 1][3] - D[i + 1] <= p[k][j] + data[j + 1][3] - D[j + 1]

            nr_customers = len(data)

            # compute cumulative distance
            demand = [0] * nr_customers
            cumulated_load = [0] * nr_customers
            duration = [0] * nr_customers
            for index in range(nr_customers):
                distance = 0
                for i in range(0, index):
                    if index > 0:
                        distance += data[i + 1][2]
                demand[index] = distance

            # compute cumulative load
            for index in range(nr_customers):
                load_temp = 0
                duration_temp = 0
                for i in range(1, index + 1):
                    load_temp += data[i][4]
                    duration_temp += data[i][5]
                cumulated_load[index] = load_temp
                duration[index] = duration_temp

            # initializing lists
            p = [[sys.maxsize for _ in range(nr_customers)] for _ in range(nr_vehicles + 1)]
            qu = Queue()
            pred = [[0 for _ in range(nr_customers)] for _ in range(nr_vehicles + 1)]
            p[0][0] = 0

            subsolution = list()
            for k in range(nr_vehicles):
                qu.queue.clear()
                qu.push_back(k)
                for t in range(k + 1, nr_customers):
                    if len(qu.queue) > 0:
                        p[k + 1][t] = p[k][qu.front()] + cost_achieved(data, i=qu.front(), x=t, Q=cumulated_load,
                                                                       D=demand,
                                                                       T=duration)
                        pred[k + 1][t] = qu.front()
                        if t < nr_customers - 1:
                            if not dominates(data=data, i=qu.back(), j=t, k=k, p=p, D=demand, Q=cumulated_load,
                                             T=duration):
                                while len(qu.queue) > 0 and dominates(data=data, i=t, j=qu.back(), k=k,
                                                                      Q=cumulated_load, p=p,
                                                                      D=demand,
                                                                      T=duration):
                                    qu.pop_back()
                                qu.push_back(t=t)
                            while len(qu.queue) > 1 and p[k][qu.front()] + cost_achieved(data=data, i=qu.front(),
                                                                                         x=t + 1,
                                                                                         Q=cumulated_load, D=demand,
                                                                                         T=duration) >= \
                                    p[k][qu.front2()] + cost_achieved(data=data, i=qu.front2(), x=t + 1,
                                                                      Q=cumulated_load, D=demand,
                                                                      T=duration):
                                qu.pop_front()

            start = pred[nr_vehicles][nr_customers - 1]
            subsolution.append([data[x][0] for x in range(start + 1, nr_customers)])
            index = start
            for i in range(nr_vehicles - 1, 0, -1):
                start = pred[i][index]
                subsolution.append([data[x][0] for x in range(start + 1, index + 1)])
                index = start
            return subsolution

        depots = list(self.gt_chromosome.keys())
        data = create_split_data(self)

        # solution form: {d_id:[[subtour],...], d_id:[[subtour,...]]}
        # nr_depots = self.vrp_data.nr_depots
        sub_solution = {}
        for index, i in enumerate(depots):
            max_tour_duration = self.vrp_data.depots[i].max_route_duration
            nr_vehicles = self.vrp_data.depots[i].nr_vehicles
            Q_max = self.vrp_data.depots[i].max_vehicle_load
            sub_solution[i] = linear_split(
                data=data[index], Q_max=Q_max, max_tour_duration=max_tour_duration, nr_vehicles=nr_vehicles)
        return sub_solution
