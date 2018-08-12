import statistics as stat
import openpyxl as xl
import numpy as np

MONITORING = False
SOLUTION_FILE = "Data-Analysis.xlsx"


class DataMonitor:
    def __init__(self):
        if MONITORING:
            self._data = {'instance': [], 'number of customers': [], 'number of vehicles': [], 'maximum length': [],
                          'load constraint': [], 'average distance of customers': [], 'number depots': []
                          }
            wb = xl.load_workbook(SOLUTION_FILE)
            sht = wb["Data"]
            sht.append(sorted(self._data.keys()))
            wb.save(SOLUTION_FILE)

    def add_info(self, data):
        if MONITORING:
            self._data['number of customers'].append(data.nr_customers)
            self._data['number of vehicles'].append(data.nr_vehicles)
            self._data['number depots'].append(data.nr_depots)
            self._data['load constraint'].append(data.depots[0].max_vehicle_load)
            max_length = data.depots[0].max_route_duration if data.depots[0].max_route_duration is not np.inf else 0
            self._data['maximum length'].append(max_length)
            self._data['instance'].append(data.dat_path[-8:])
            average_customer_distance = stat.mean([stat.mean(x) for x in data.distance_matrix])
            self._data['average distance of customers'].append(average_customer_distance)



    def write(self):
        if MONITORING:
            info = []
            for key in sorted(self._data.keys()):
                try:
                    info.append(self._data[key].pop())
                except IndexError:
                    info.append('n/a')

            wb = xl.load_workbook(SOLUTION_FILE)
            sht = wb["Data"]
            sht.append(info)
            wb.save(SOLUTION_FILE)
