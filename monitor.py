import statistics as stat
# import openpyxl as xl
import datetime

MONITORING = False

class Monitor:
    def __init__(self, dataset):
        if MONITORING:
            self._data = {'iteration': [],
                          # population
                          'time_elapsed': [], 'pop-size': [], 'pop-best': [], 'pop-mean-fitness-feasible': [],
                          'pop-mean-fitness-unfeasible': [], 'load-penalty': [], 'duration-penalty': [],
                          'entropy': [], 'sum-list-quantity': [], 'sum-list-distance': [],
                          # parent sets
                          'parent-mean-length': [],
                          # children
                          'child-fitness': [], 'child-feasible': [],
                          'child-fitness-after-education-and-repair': []
                          }

            self._wb = xl.Workbook()
            self._sht = self._wb.active

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self._file = "monitor-%s-%s.xlsx" % (dataset, timestamp)
            self._sht.append(sorted(self._data.keys()))
            self._wb.save(self._file)
            self._iter = 1

    def evaluation_population(self, feasible_pop, unfeasible_pop, time_elapsed):
        if MONITORING:
            self._data['time_elapsed'].append(time_elapsed)
            self._data['pop-size'].append(len(feasible_pop) + len(unfeasible_pop))
            # self._data['pop-worst'].append(min(members).fitness)
            if len(feasible_pop) > 0:
                mean_fitness_feasible = stat.mean([x.length for x in feasible_pop])
                self._data['pop-mean-fitness-feasible'].append(mean_fitness_feasible)
            if len(feasible_pop) > 0:
                mean_fitness_infeasible = stat.mean([x.length for x in unfeasible_pop])
                self._data['pop-mean-fitness-unfeasible'].append(mean_fitness_infeasible)
            # self._data['pop-mean-distance'].append(mean_distance(members))

    def evaluate_offspring(self, child):
        if MONITORING:
            self._data['child-fitness'].append(child.length)
            self._data['child-feasible'].append(child.feasibility)

    def evaluate_offspring_after_education(self, child):
        if MONITORING:
            self._data['child-fitness-after-education-and-repair'].append(child.length)

    def evaluate_feasibility(self, list_quantity, list_distance):
        if MONITORING:
            self._data['sum-list-distance'].append(sum(list_distance))
            self._data['sum-list-quantity'].append(sum(list_quantity))

    def evaluate_best(self, individual):
        if MONITORING and individual is not None:
            self._data['pop-best'].append(individual.length)

    def evaluate_parents(self, parent1, parent2):
        if MONITORING:
            mean = stat.mean([parent1.length, parent2.length])
            self._data['parent-mean-length'].append(mean)

    def evaluate_penaly_factors(self, len_pen, load_pen):
        if MONITORING:
            self._data['load-penalty'].append(load_pen)
            self._data['duration-penalty'].append(len_pen)

    def evaluate_entropy(self, hamming_matrix):
        if MONITORING:
            mean = stat.mean([stat.mean(x) for x in hamming_matrix])
            self._data['entropy'].append(mean)

    def write_row(self):
        if MONITORING:
            self._data['iteration'].append(self._iter)
            info = []
            for key in sorted(self._data.keys()):
                try:
                    info.append(self._data[key].pop())
                except IndexError:
                    info.append('n/a')
            self._sht.append(info)
            self._wb.save(self._file)
            self._iter += 1


