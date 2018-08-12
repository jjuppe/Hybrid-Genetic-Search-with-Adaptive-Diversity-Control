class Node:
    def __init__(self, n_id, x, y):
        self.n_id = n_id
        self.x = x
        self.y = y


class Depot(Node):
    def __init__(self, n_id, x, y, nr_vehicles, max_route_duration, max_vehicle_load):
        """
        Constructor of a typical Depot object in a MDPVRP
        Note: No heterogeneous fleet

        Args:
            n_id: Node ID
            x: x-coordinate
            y: y-coordinate
            nr_vehicles: nr of vehicles on the depot
            max_route_duration: max route duration of each vehicle (e.g. gas tank)
            max_vehicle_load: max load of a vehicle
        """

        super().__init__(n_id, x, y)
        self.nr_vehicles = nr_vehicles
        self.max_route_duration = max_route_duration
        self.max_vehicle_load = max_vehicle_load


class Customer(Node):
    def __init__(self, n_id, x, y, service_duration, demand, frequency_of_visit, len_visit_comb, list_visit_comb):
        """
        Standard constructor of a customer node in a MDPVRP setting.
        Note: No time windows

        Args:
            n_id: Node ID
            x: x-coordinate
            y: y-coordinate
            service_duration: Duration to serve a customer
            demand: Demand of the customer
            frequency_of_visit: Demanded visit frequency in the given time window
            len_visit_comb: Indicator of len list_visit_comb
            list_visit_comb: Indicating all windows that are allowed for visit
        """

        super().__init__(n_id, x, y)
        self.service_duration = service_duration
        self.demand = demand
        self.frequency_of_visit = frequency_of_visit
        self.len_visit_comb = len_visit_comb
        self.list_visit_comb = list_visit_comb