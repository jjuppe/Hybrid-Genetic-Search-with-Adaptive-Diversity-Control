import individual
import math
import logging
import numpy as np
import random
import sys
from time import time
from individual import Individual
from tools import get_hemming_distance_matrix, normalized_hamming_distance
from tools import remove_column
from tools import get_rank_array
from plotter import plotter
import functools
import itertools
import copy
from monitor import Monitor



# setting up different logger
GENERAL_LOGGER = logging.getLogger("general_info")
POPULATON_LOGGER = logging.getLogger("population_management")
CROSSOVER_LOGGER = logging.getLogger("crossover")

# set logger level
GENERAL_LOGGER.setLevel(logging.INFO)
POPULATON_LOGGER.setLevel(logging.INFO)
CROSSOVER_LOGGER.setLevel(logging.WARNING)

# initialized general logging
logging.getLogger().setLevel(logging.INFO)
logging.info("Start Genetic Algorithm")

HAMMING_DISTANCE_MATRIX = []  # hemming distance matrix of current population


def merge_population(population):
    """
    create a merged population representation

    """

    return population["feasible"] + population["infeasible"]


def branch_population(merged_population):
    """
    create an unmerged population representation

    """

    feasible_pop = []
    infeasible_pop = []
    for instance in merged_population:
        if instance.feasibility:
            feasible_pop.append(instance)
        else:
            infeasible_pop.append(instance)

    return {"feasible": feasible_pop, "infeasible": infeasible_pop}


def get_average_solution_length(population):
    """
    Calculate the average distance of the instances within a population

    Args:
        population: list of indices with format {"feasible":[], "infeasible":[]}

    Returns: avg distance of instances in the population

    """
    sum_length = 0
    nr_instances = 0
    for pop_type in population:
        sub_pop = population[pop_type]
        nr_instances += len(sub_pop)

        for instance in sub_pop:
            sum_length += instance.length
    return sum_length / nr_instances


def find_clones(population):
    """
    Find all individuals that have a clone.
    A clone is:

    1) find all individuals with pattern and depot assignment
    2) find all individuals with same cost

    However, there is the chance that there is a clone with identical depot_chromosome but different solution.
    This is because of the solution of the split algorithm and the crossover algorithm is not identical
    Therefore, if this happens, we select the version with the shortest length.

    Args:
        population: List of individuals

    Returns: List of individuals

    """
    tmp_pop = population.copy()
    clones = []

    while tmp_pop:
        individual1 = tmp_pop[0]
        clone_found = False

        for i, individual2 in enumerate(tmp_pop[1:]):
            # if its a clone, remove the worse clone
            szen1 = (individual1.depot_chromosome == individual2.depot_chromosome)
            szen2 = (individual1.length == individual2.length)
            if szen1 or szen2:
                if individual1.length > individual2.length:
                    index_pop = 0
                else:
                    index_pop = i + 1

                worst_clone = tmp_pop.pop(index_pop)
                clones.append(worst_clone)
                clone_found = False
                break

        # else remove the first object
        if not clone_found:
            del tmp_pop[0]
    return clones


def get_maximum_fitness_instance(merged_population):
    """
    small helper function returning the instance of a population with maximum fitness

    """

    max_fit_inst = None
    max_biased_fitness = 0

    for inst in merged_population:
        if max_biased_fitness < inst.fitness:
            max_fit_inst = inst
            max_biased_fitness = inst.fitness
    return max_fit_inst


def get_maximum_fitness_instance_index(merged_population):
    """
    small helper function returning the instance of a population with maximum fitness

    """

    index = 0
    max_biased_fitness = 0

    for i, inst in enumerate(merged_population):
        if max_biased_fitness < inst.fitness:
            index = i
            max_biased_fitness = inst.fitness

    return index


def get_minimum_length_instance(merged_population):
    """
    small helper function to return the instance with minimum length

    """

    min_length_inst = None
    min_length = np.inf

    for inst in merged_population:
        if min_length > inst.length:
            min_length_inst = inst
            min_length = inst.length
    return min_length_inst


def extend_hamming_distance_matrix(population, offspring):
    """
    Extend an existing hemming distance_matrix

    """
    global HAMMING_DISTANCE_MATRIX

    dist_arr = [normalized_hamming_distance(offspring, inst2) for inst2 in population]
    dist_arr += [0]

    for i, row in enumerate(HAMMING_DISTANCE_MATRIX):
        row.append(dist_arr[i])
    HAMMING_DISTANCE_MATRIX.append(dist_arr)
    return HAMMING_DISTANCE_MATRIX


class MDCPVRPSolution:
    """

    The following parameters can be adjusted:
        - vrp_data:
        - n_close_factor:

    """
    def __init__(self, vrp_data, min_sub_pop_size=25, population_size=100,
                 n_close_factor=0.11, target_proportion=0.36, nr_elite_individuals=0.4, max_iter_wo_improvement=10000,
                 iter_diversification=0.6, repair_probability=0.7, education_probability=0.9,
                 iter_penalty_adjust=100, max_time=600):
        self.vrp_data = vrp_data
        self.min_sub_pop_size = min_sub_pop_size
        self.population_size = population_size
        self.n_close = math.floor(min_sub_pop_size * n_close_factor)
        self.target_proportion = target_proportion
        self.nr_elite_individuals = math.floor(min_sub_pop_size * nr_elite_individuals)
        self.w_penalty_duration = 1
        self.w_penalty_load = 1
        self.repair_probability = repair_probability
        self.education_probability = education_probability
        self.max_iter_wo_improvement = max_iter_wo_improvement
        self.iter_diversification = math.ceil(max_iter_wo_improvement * iter_diversification)
        self.iter_penalty_adjust = iter_penalty_adjust
        self.max_time = max_time

        # those are the solution values
        self.solution = None

    def solve(self):
        """
        Population management and solving structure of the genetic heuristic.

        Includes:
        1) Initialization of population
        2) Initialization of weights
        3) Weight adjustment
        4) Crossover (w. Parent selection)
        5) Diversification

        After the set stoppage criterion is met the solution variable will be allocated

        """
        global HAMMING_DISTANCE_MATRIX

        # 0) LOCAL SAVE STATIC PARAMETERS
        target_proportion = self.target_proportion
        nr_customers = self.vrp_data.nr_customers
        max_size = self.population_size + self.min_sub_pop_size
        iter_penalty_adjust = self.iter_penalty_adjust

        avg_demand = sum([customer for customer in self.vrp_data.customers]) / nr_customers
        diversification_size = math.floor(self.min_sub_pop_size / 3)

        # 1) initialize the first population and set population specific parameters
        population = self.random_population_initialization()
        list_feas_quantity = [False] * iter_penalty_adjust
        list_feas_distance = [False] * iter_penalty_adjust
        inf_pop_size = len(population["infeasible"])
        feas_pop_size = len(population["feasible"])

        # 2) initialize weights
        # get average distance between customers (length of solution / nr_customers)
        # we must add 1 as the distance from and to the depot is also considered c+1 arcs
        avg_distance_customers = get_average_solution_length(population)
        self.w_penalty_load = avg_distance_customers / avg_demand

        # 3) Do calculations
        try:
            best_instance = get_minimum_length_instance(population["feasible"])
            best_instance_length = best_instance.length
        except AttributeError:
            best_instance = None
            best_instance_length = np.inf  # infinite -> no valid solution found so far
            GENERAL_LOGGER.warning("No feasible solution in the initial solution set")

        # initialize stoppage criteria
        max_iter_wo_improvement, iter_without_impr, iter_wa = self.max_iter_wo_improvement, 0, 0
        max_time, start_t = self.max_time, time()
        monitor = Monitor("MDVRP" + str(self.vrp_data.nr_customers))
        while time() - start_t < max_time and iter_without_impr < max_iter_wo_improvement:
            # 3.1) PARENT SELECTION
            parent_1 = self.parent_selection(population)
            parent_2 = self.parent_selection(population)
            monitor.evaluate_parents(parent_1, parent_2)

            # 3.2) CROSSOVER (PIX)
            offspring = self.crossover(parent_1, parent_2)
            monitor.evaluate_offspring(offspring)

            # 3.3) EDUCATION
            length_pen_factor = self.w_penalty_duration
            load_pen_factor = self.w_penalty_load
            monitor.evaluate_penaly_factors(length_pen_factor, load_pen_factor)

            all_offsprings = [offspring]
            if random.random() <= self.education_probability:
                offspring.route_improvement(load_pen_factor, length_pen_factor)
                offspring.pattern_improvement()
                offspring.route_improvement(load_pen_factor, length_pen_factor)
                if not offspring.feasibility and random.random() <= self.repair_probability:
                    repair_offspring = copy.deepcopy(offspring)
                    for x in range(1, 3):
                        length_pen_factor *= 10
                        load_pen_factor *= 10
                        repair_offspring.route_improvement(load_pen_factor, length_pen_factor)
                        repair_offspring.pattern_improvement()
                        repair_offspring.route_improvement(load_pen_factor, length_pen_factor)
                        repair_offspring.evaluate_solution()
                        if repair_offspring .feasibility:
                            all_offsprings.append(repair_offspring)
                            continue

            monitor.evaluate_offspring_after_education(all_offsprings[0])

            for offspring in all_offsprings:
                list_feas_quantity[iter_wa] = (offspring.penalty_duration == 0)
                list_feas_distance[iter_wa] = (offspring.penalty_load == 0)
                monitor.evaluate_feasibility(list_feas_quantity, list_feas_distance)

                # 3.4) APPEND HEMMING DISTANCE
                merged_population = merge_population(population)
                hamming_distance_matrix = extend_hamming_distance_matrix(merged_population, offspring)
                monitor.evaluate_entropy(hamming_distance_matrix)

                # for offspring in all_offspring:
                if offspring.feasibility:
                    # FEASIBLE OFFSPRING HANDLING
                    feas_pop_size += 1
                    population["feasible"].append(offspring)

                    # check if we need to perform survivor selection
                    if feas_pop_size > max_size:
                        POPULATON_LOGGER.debug("feasible_survivor_selection")
                        feas_pop_size = self.min_sub_pop_size
                        self.survivor_selection(population, "feasible", self.min_sub_pop_size)

                    # check if we improved the best solution
                    if best_instance_length > offspring.length:
                        best_instance = offspring
                        best_instance_length = offspring.length

                        GENERAL_LOGGER.info(best_instance_length)
                        iter_without_impr = 0
                    else:
                        iter_without_impr += 1
                else:
                    # INFEASIBLE OFFSPRING HANDLING
                    inf_pop_size += 1
                    population["infeasible"].append(offspring)

                    # check if we need to perform survivor selection
                    if inf_pop_size > max_size:
                        POPULATON_LOGGER.debug("infeasible_survivor_selection")
                        inf_pop_size += self.min_sub_pop_size
                        self.survivor_selection(population, "infeasible", self.min_sub_pop_size)
                    iter_without_impr += 1

                # 3.4) WEIGHT ADJUSTMENTS EVERY iter_wa ITERATIONS
                if iter_wa == (iter_penalty_adjust - 1):
                    iter_wa = 0
                    feasibility_proportion_quantity = sum(list_feas_quantity) / iter_penalty_adjust
                    feasibility_proportion_distance = sum(list_feas_distance) / iter_penalty_adjust

                    if feasibility_proportion_quantity <= target_proportion - 0.05:
                        self.w_penalty_load = self.w_penalty_load * 1.2
                    elif feasibility_proportion_quantity >= target_proportion - 0.05:
                        self.w_penalty_load = self.w_penalty_load * 0.85

                    if feasibility_proportion_distance <= target_proportion - 0.05:
                        self.w_penalty_duration = self.w_penalty_duration * 1.2
                    elif feasibility_proportion_distance >= target_proportion - 0.05:
                        self.w_penalty_duration = self.w_penalty_duration * 0.85

                else:
                    iter_wa += 1

                # 3.5) DIVERSIFICATION
                if iter_without_impr > 0 and (iter_without_impr % self.iter_diversification) == 0:
                    GENERAL_LOGGER.info("diversification")
                    # the length of both population subsets will be reduced
                    feas_pop_size, inf_pop_size = diversification_size, diversification_size
                    # we diversify if the iterations without improvement is a true dividor of the target
                    population = self.survivor_selection(population, "feasible", diversification_size)
                    population = self.survivor_selection(population, "infeasible", diversification_size)
                    # perform random fill
                    population = self._random_fill_population(population)

            monitor.evaluation_population(population['feasible'], population['infeasible'], time() - start_t)
            monitor.evaluate_best(best_instance)
            monitor.write_row()

        self.solution = best_instance

    def random_population_initialization(self):
        """
        Initialize population by the randomized scheme
        Note: At the end of the initialization one sub-population might be incomplete (allowed)

        Returns: population {"feasible":feasible_sols, "infeasible":infeasible_sols}

        """

        population = {"feasible": [], "infeasible": []}
        population = self._random_fill_population(population)
        return population

    def _random_fill_population(self, population):
        """
        Generate 4*mu random instances. If the size of the population exceeds a certain limit remove the worst parts

        Args:
            population: Population filled with the best random solutions

        Returns: Stuffed population

        """
        global HAMMING_DISTANCE_MATRIX

        vrp_data = self.vrp_data
        min_sub_pop_size = self.min_sub_pop_size
        max_size = self.population_size + min_sub_pop_size

        len_fs, len_ifs = len(population["feasible"]), len(population["infeasible"])
        for i in range(self.min_sub_pop_size * 4):
            new_sol = individual.random_instance(vrp_data, self)

            # check if the solution is feasible
            if new_sol.feasibility:
                population["feasible"].append(new_sol)
                len_fs += 1
                if len_fs > max_size:
                    len_fs = self.min_sub_pop_size
                    HAMMING_DISTANCE_MATRIX = get_hemming_distance_matrix(merge_population(population))
                    population = self.survivor_selection(
                        population, "feasible", self.min_sub_pop_size)
            else:
                population["infeasible"].append(new_sol)
                len_ifs += 1
                if len_ifs > max_size:
                    len_ifs = self.min_sub_pop_size
                    HAMMING_DISTANCE_MATRIX = get_hemming_distance_matrix(merge_population(population))
                    population = self.survivor_selection(
                        population, "infeasible", self.min_sub_pop_size)

        # finally set the hemming distance matrix
        HAMMING_DISTANCE_MATRIX = get_hemming_distance_matrix(merge_population(population))
        return population

    def survivor_selection(self, population, sub_pop_type, nr_survivors):
        """
        Perform survivor selection one one sub-population to keep only the fittest
        DISCLAIMER: Not the prettiest implementation.... i know i know

        Args:
            population:
            sub_pop_type: type of sub-population ("feasible" / "infeasible")
            nr_survivors: Number of instances that should survive the elimination

        Returns:

        """
        global HAMMING_DISTANCE_MATRIX  # get the hemming_distance_matrix
        t = HAMMING_DISTANCE_MATRIX
        # find clones of only the relevant population type
        clones = find_clones(population[sub_pop_type])
        merged_pop = merge_population(population)

        pop_size = len(population[sub_pop_type])
        while pop_size > nr_survivors:
            # calculate the fitness based on full population
            self.set_biased_population_fitness(population)

            # 1) Remove clones (worst fitness clones first)
            if clones:
                # find minimum fitness
                clone_index = get_maximum_fitness_instance_index(clones)
                indiv = clones[clone_index]
                index = population[sub_pop_type].index(indiv)
                individual_index = merged_pop.index(indiv)
                del clones[clone_index]
                del population[sub_pop_type][index]
                del merged_pop[individual_index]
            else:
                # 2) Remove worst fitness instances after no clones exist
                index = get_maximum_fitness_instance_index(population[sub_pop_type])
                individual_index = merged_pop.index(population[sub_pop_type][index])
                del population[sub_pop_type][index]
                del merged_pop[individual_index]

            del HAMMING_DISTANCE_MATRIX[individual_index]  # remove column
            remove_column(HAMMING_DISTANCE_MATRIX, individual_index)  # remove distance

            pop_size -= 1

        return population

    def set_biased_population_fitness(self, population):
        """
        Get the biased fitness of all individuals within the population

        Args:
            population: list of instances (prev. merged)

        Returns:

        """

        # it is important to note that we calculate the diversity ranking on the basis of the whole population
        # However, the length ranking must be performed on subpopulation basis, in order to ensure, that the elite
        # solutions survive
        merged_population = merge_population(population=population)
        diversity_array = np.array(self.get_diversity_contribution_array(merged_population))

        # get the ranks of each entry (descending = High is good)
        div_rank = get_rank_array(diversity_array, type = "desc")

        # we consider the diversity within the whole population (other than the paper)
        # Therefore, we need to normalize the rank and round the rank up

        # get the quality values of each individual
        # 0) SET STANDARD PARAMETERS
        len_total = len(merged_population)

        # 1) COMPUTE FITNESS FOR FEASIBLE SOLUTIONS
        len_feas = len(population["feasible"])
        len_infeas = len(population["infeasible"])

        quality_array = np.array([self.get_solution_quality(indiv) for indiv in population["feasible"]])
        quality_ranking = get_rank_array(quality_array, type = "asc")  # ascending, low is good

        # if no feasible solution was found, skip this part
        if len_feas > 0:
            diversity_w = (1 - self.nr_elite_individuals / len_feas)
            for i, inst in enumerate(population["feasible"]):
                div_rank_inst = math.ceil(div_rank[i] * (len_feas / len_total))
                inst.fitness = quality_ranking[i] + diversity_w * div_rank_inst

        # 2) DO THE SAME FOR THE INFEASIBLE SOLUTIONS
        if len_infeas > 0:
            quality_array = np.array([self.get_solution_quality(indiv) for indiv in population["infeasible"]])
            quality_ranking = get_rank_array(quality_array, type = "asc")  # ascending, low is good

            diversity_w = (1 - self.nr_elite_individuals / len_infeas)
            for i, inst in enumerate(population["infeasible"]):
                # important, here + len_feas to the index because the diversity array continues
                div_rank_inst = math.ceil(div_rank[i + len_feas] * (len_infeas / len_total))
                # important, here i as an index because we recalculate the quality array
                inst.fitness = quality_ranking[i] + diversity_w * div_rank_inst

    def get_diversity_contribution_array(self, merged_population):
        """
        Get an array containing all diversity contribution values of each individual in the population
        Args:
            merged_population: combined population of instances in one list

        Returns: array

        """
        global HAMMING_DISTANCE_MATRIX

        # get the diversity scores of each individual
        diversity_array = []
        n_close = self.n_close

        for i in range(len(merged_population)):
            t = HAMMING_DISTANCE_MATRIX
            try:
                dist_arr = HAMMING_DISTANCE_MATRIX[i].copy()
            except IndexError:
                print(None)
                raise IndexError("Stuff")

            # get the n nearest neightbors
            dist_arr.sort()
            diversity_cont_instance = sum(
                dist_arr[1:n_close + 1]) / n_close  # the first element will always be the object itself

            diversity_array.append(diversity_cont_instance)

        return diversity_array

    def get_solution_quality(self, solution):
        """
        get the quality of a solution based on the penalty weights set in our problem instance

        """
        return solution.length + solution.penalty_duration * self.w_penalty_duration + solution.penalty_load * self.w_penalty_load



    def parent_selection(self, population):
        """
        Binary tournament for parent selection.
        Allowes to compare one parent with itself, but that should not impact the solution quality too much

        Args:
            population:

        Returns:

        """

        # 1) set most recent fitness values
        self.set_biased_population_fitness(population)

        # 2) select two candidate parents
        # get random subset
        parents = []
        w1 = len(population["feasible"])
        w2 = len(population["infeasible"])
        keys = list(population.keys())

        for i in range(2):  # two parents
            sub_pop = random.choices(keys, [w1, w2])[0]
            parents.append(random.choice(population[sub_pop]))

        # 3) select the best parent
        if parents[0].fitness < parents[1].fitness:
            return parents[0]
        else:
            return parents[1]

    def crossover(self, p1, p2):
        """
        main function to perform the PIX crossover procedure
        Args:
            p1: parent one of type instance
            p2:  parent two of type instance

        Returns:

        """
        (lambda1, lambda2, lambdaMix) = self.determine_inheritance_material(p1, p2)
        (child_gt_chromosome, child_depot_chromosome) = self.inherit_from_parent1(lambda1, lambdaMix)
        (child_gt_chromosome, child_depot_chromosome) = self.inherit_from_parent2(
            child_gt_chromosome, child_depot_chromosome, lambda2, lambdaMix, p1.vrp_data.depots)

        child = self.add_unserved_custs(child_gt_chromosome, child_depot_chromosome, p1.vrp_data)

        return child

    def determine_inheritance_material(self, p1, p2):
        nr_depots = len(p1.gt_chromosome)
        rand1 = random.randint(0, nr_depots)
        rand2 = random.randint(rand1, nr_depots)
        lambda1 = {}
        lambda2 = {}
        lambdaMix = {}
        # create set of depot indices and place it in shuffled order to a list
        setOfDepots = set(p1.gt_chromosome.keys())
        listOfDepots = random.sample(setOfDepots, len(setOfDepots))
        # take giant tours from all depots from p1 or p2 and place to lambda dicts
        for depot_index in range(rand1):
            lambda1[listOfDepots[depot_index]] = p1.gt_chromosome[listOfDepots[depot_index]]
        for depot_index in range(rand1, rand2):
            lambda2[listOfDepots[depot_index]] = p2.gt_chromosome[listOfDepots[depot_index]]
        for depot_index in range(rand2, nr_depots):
            if (bool(random.getrandbits(1))):
                lambdaMix[listOfDepots[depot_index]] = p1.gt_chromosome[listOfDepots[depot_index]]
            else:
                lambdaMix[listOfDepots[depot_index]] = p2.gt_chromosome[listOfDepots[depot_index]]
        CROSSOVER_LOGGER.debug("nr_depots: %d", nr_depots)
        CROSSOVER_LOGGER.debug("lambda1: %", lambda1)
        CROSSOVER_LOGGER.debug("lambda2: %f", lambda2)
        CROSSOVER_LOGGER.debug("lambdaMix: %s", lambdaMix)
        return (lambda1, lambda2, lambdaMix)

    # inherit data from P1
    def inherit_from_parent1(self, lambda1, lambdaMix):
        """creates a giant tour chromosome using sets lambda1 and lambdaMix

        Adds the customer out of set lamda1 to the giant tour chromosome to of the child.
        Then adds customers from set lambdaMix as well making sure that no duplicate notes are inserted.

        Args:
            lambda1: random set of customers from parent 1 given in a dictionary with depots as keys
            lambdaMix: random set of customers from both parents, given in a dictionary with depots as keys

        Returns: an incomplete giant tour chromosome and depot chromosome

        """
        child_gt_chromosome = {}
        child_depot_chromosome = {}
        # add everything from lambda1 to child
        child_gt_chromosome.update(lambda1)

        for depot, tour in lambdaMix.items():
            if len(tour) == 0:
                continue
            rand1 = random.randint(0, len(tour) - 1)
            rand2 = random.randint(rand1, len(tour) - 1)
            # if depot does not exist yet
            if depot not in child_gt_chromosome:
                child_gt_chromosome[depot] = list()
            for customer in lambdaMix[depot][rand1:rand2]:
                all_cust = [x for dep, x in child_gt_chromosome.items()]
                all_cust_as_list = [j for sub in all_cust for j in sub]
                if customer not in all_cust_as_list:
                    CROSSOVER_LOGGER.debug("GT: added %d to %d", customer, depot)
                    child_gt_chromosome[depot].append(customer)

        # normal duplicate check
        duplicate_check = set()
        for depot, tour in child_gt_chromosome.items():
            for cust in tour:
                if cust in duplicate_check:
                    # print('Duplicate:' + str(cust))
                    tour.remove(cust)
                else:
                    duplicate_check.add(cust)

        all_cust = [x for dep, x in child_gt_chromosome.items()]
        all_cust_as_list = [j for sub in all_cust for j in sub]
        dup = set()
        for n in all_cust_as_list:
            if n in dup:
                for depot, tour in child_gt_chromosome.items():
                    if n in tour:
                        tour.remove(n)
                        break
            else:
                dup.add(n)

        for depot, tour in child_gt_chromosome.items():
            for cust in tour:
                CROSSOVER_LOGGER.debug("depot: added %d to %d", depot, cust)
                child_depot_chromosome[cust] = [depot]

        CROSSOVER_LOGGER.debug("")
        CROSSOVER_LOGGER.debug("Step1")
        CROSSOVER_LOGGER.debug("child.gt_chromosome: %s", child_gt_chromosome)
        CROSSOVER_LOGGER.debug("child.depot_chromosome: %s", child_depot_chromosome)

        return child_gt_chromosome, child_depot_chromosome

    # inherit data from P2
    def inherit_from_parent2(self, child_gt_chromosome, child_depot_chromosome, lambda2, lambdaMix, depots):
        lambdaNew = {}
        lambdaNew.update(lambda2)
        lambdaNew.update(lambdaMix)
        CROSSOVER_LOGGER.debug("")
        CROSSOVER_LOGGER.debug("")
        keys = list(lambdaNew.keys())
        random.shuffle(keys)
        # i is depot, j is customer
        for depot_index in keys:
            for cust_index in lambdaNew[depot_index]:
                # second condition is meaningless in single period MDVRP
                # if(j not in child.depot_chromosome or child.depot_chromosome[j][0] == i):
                if cust_index not in child_depot_chromosome:
                    if depot_index in child_gt_chromosome:
                        CROSSOVER_LOGGER.debug("GT: %d existed, added %d", depot_index, cust_index)
                        child_gt_chromosome[depot_index].append(cust_index)
                    else:
                        CROSSOVER_LOGGER.debug("GT: %d didn't exist, added, %d", depot_index, cust_index)
                        child_gt_chromosome[depot_index] = [cust_index]
                    CROSSOVER_LOGGER.debug("depot: added %d to %d", [depot_index], cust_index)
                    child_depot_chromosome[cust_index] = [depot_index]

        CROSSOVER_LOGGER.debug("")
        CROSSOVER_LOGGER.debug("Step2")
        CROSSOVER_LOGGER.debug("child.gt_chromosome: %s", child_gt_chromosome)
        CROSSOVER_LOGGER.debug("child.depot_chromosome: %s", child_depot_chromosome)

        # add all other depots to gt chromosome and
        for depot_index in depots:
            if depot_index not in child_gt_chromosome:
                child_gt_chromosome[depot_index] = []

        return child_gt_chromosome, child_depot_chromosome

    def add_unserved_custs(self, child_gt_chromosome, child_depot_chromosome, vrp_data):

        CROSSOVER_LOGGER.debug("")
        CROSSOVER_LOGGER.debug("")
        CROSSOVER_LOGGER.debug("Step3")
        child = Individual(vrp_data, self, child_depot_chromosome, child_gt_chromosome)
        # create a list of unvisited customers
        unvisited_custs = []
        all_custs = vrp_data.customers.keys()
        CROSSOVER_LOGGER.debug("Before: %s", child.solution)
        for cust_index in all_custs:
            if (cust_index not in child_depot_chromosome):
                unvisited_custs.append(cust_index)
        random.shuffle(unvisited_custs)
        CROSSOVER_LOGGER.debug("Unvisited customers: %s", unvisited_custs)

        dist_matr = vrp_data.distance_matrix
        cust_data = vrp_data.customers

        for cust_index in unvisited_custs:
            # initialize
            CROSSOVER_LOGGER.debug("cust_index = %d", cust_index)
            least_insertion_cost = sys.maxsize
            best_depot_index = None
            best_tour_index = None
            best_after_cust = None
            for depot_index in child_gt_chromosome:
                CROSSOVER_LOGGER.debug("depot_index = %d", depot_index)

                for ti, tour in enumerate(child.solution[depot_index]):
                    CROSSOVER_LOGGER.debug("ti = %f", ti)
                    # add customer to tour
                    if len(tour) < 1:
                        length_increase = dist_matr[depot_index][cust_index] + \
                                          dist_matr[cust_index][depot_index]
                    else:
                        length_increase = dist_matr[depot_index][cust_index] + \
                                          dist_matr[cust_index][tour[0]] - dist_matr[depot_index][tour[0]]

                    CROSSOVER_LOGGER.debug("child.sub_tour_lengths[depot_index] = %s",
                                           child.sub_tour_lengths[depot_index])
                    CROSSOVER_LOGGER.debug("ti = %d", ti)
                    child.sub_tour_lengths[depot_index][ti] += length_increase
                    child.sub_tour_loads[depot_index][ti] += cust_data[cust_index].demand
                    total_increase = length_increase + self.cumulated_penalty_values(
                        depot_index, child.sub_tour_lengths[depot_index][ti],
                        child.sub_tour_service_times[depot_index][ti],
                        child.sub_tour_loads[depot_index][ti])
                    # check if the best solution position is after the depot
                    # CROSSOVER_LOGGER.info("Least Insertion Cost %f", least_insertion_cost)
                    least_insertion_cost = total_increase
                    best_depot_index = depot_index
                    best_tour_index = ti
                    best_after_cust = -1  # position is in the beginning

                    # remove customer from tour
                    child.sub_tour_lengths[depot_index][ti] -= length_increase
                    child.sub_tour_loads[depot_index][ti] -= cust_data[cust_index].demand

                    for aci, after_cust in enumerate(tour):
                        CROSSOVER_LOGGER.debug("aci = %d", aci)
                        CROSSOVER_LOGGER.debug("after_cust = %s", after_cust)
                        # add customer to tour
                        # if we are checking the last element of a tour
                        if aci == len(tour) - 1:
                            length_increase = dist_matr[after_cust][cust_index] + \
                                              dist_matr[cust_index][depot_index] - \
                                              dist_matr[after_cust][depot_index]
                        else:
                            length_increase = dist_matr[after_cust][cust_index] + \
                                              dist_matr[cust_index][tour[aci + 1]] - \
                                              dist_matr[after_cust][tour[aci + 1]]
                        child.sub_tour_lengths[depot_index][ti] += length_increase
                        child.sub_tour_loads[depot_index][ti] += cust_data[cust_index].demand
                        total_increase = length_increase + self.cumulated_penalty_values(depot_index,
                                                                                         child.sub_tour_lengths[
                                                                                                    depot_index][ti],
                                                                                         child.sub_tour_service_times[depot_index][ti],
                                                                                         child.sub_tour_loads[
                                                                                                    depot_index][ti])

                        # check if best insertion position is after aci
                        if total_increase < least_insertion_cost:
                            # CROSSOVER_LOGGER.info("Least Insertion Cost %f", least_insertion_cost)

                            least_insertion_cost = total_increase
                            best_depot_index = depot_index
                            best_tour_index = ti
                            best_after_cust = aci

                        # remove customer from tour
                        child.sub_tour_lengths[depot_index][ti] -= length_increase
                        child.sub_tour_loads[depot_index][ti] -= cust_data[cust_index].demand

            # insert customer to chromosomes
            child.gt_chromosome[best_depot_index].append(cust_index)
            child.depot_chromosome[cust_index] = [best_depot_index]

            # insert customer to solution
            child.solution[best_depot_index][best_tour_index] = child.solution[best_depot_index][best_tour_index][
                                                                0:best_after_cust + 1] + [
                                                                    cust_index] + child.solution[best_depot_index][
                                                                                      best_tour_index][
                                                                                  best_after_cust + 1:]

            CROSSOVER_LOGGER.debug("Inserted customer %d to depot %d tour %d",
                                   cust_index, best_depot_index, best_tour_index)

        # add length increases to solution
        child.evaluate_solution()
        CROSSOVER_LOGGER.debug("After: %s", child.solution)
        return child

    def cumulated_penalty_values(self, depot_index, sub_tour_length, sub_tour_service_time, sub_tour_load):
        depot = self.vrp_data.depots[depot_index]
        max_route_duration = depot.max_route_duration
        max_vehicle_load = depot.max_vehicle_load
        omega_duration = max(0, sub_tour_length + sub_tour_service_time - max_route_duration) * self.w_penalty_duration
        omega_load = max(0, sub_tour_load - max_vehicle_load) * self.w_penalty_load

        return omega_duration + omega_load