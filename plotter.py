import matplotlib.pyplot as plt
import logging


def test():
    x = [-1, 0.5, 1, -0.5]
    y = [0.5,  1, -0.5, -1]
    # fig = plt.figure(figsize=(26, 18))
    # ax = plt.axes(projection=ccrs.PlateCarree())

    # plt.plot(x, y, 'ro')
    for i in range(0, len(x)):
        plt.plot(x[i:i+2], y[i:i+2], 'rx-')
    plt.show()


def plotter(individual):

    # iterate over depots
    depot_ids = list(individual.solution.keys())

    for depot_index in depot_ids:
        # print(type(individual.vrp_data.depots[depot_index].x))
        plt.plot(individual.vrp_data.depots[depot_index].x,
                 individual.vrp_data.depots[depot_index].y, 'ro')
        # iterate over tours from this depot
        for index, tour in enumerate(individual.solution[depot_index]):
            colors = ['y', 'g', 'm', 'b', 'r', 'r']
            last_cust = depot_index
            for cust_index in tour:
                # logging.info(cust_index)
                plt.plot(individual.vrp_data.customers[cust_index].x,
                         individual.vrp_data.customers[cust_index].y, 'bx')

                # plt.annotate(
                # cust_index, (individual.vrp_data.customers[cust_index].x, individual.vrp_data.customers[cust_index].y), (-20, 20))
                plt.annotate(
                    cust_index,
                    xy=(individual.vrp_data.customers[cust_index].x,
                        individual.vrp_data.customers[cust_index].y), xytext=(-3, 3),
                    textcoords='offset points', ha='right', va='bottom')
                # plot depot to first customer
                if last_cust < individual.vrp_data.nr_depots:
                    plt.plot([individual.vrp_data.customers[cust_index].x, individual.vrp_data.depots[last_cust].x], [
                             individual.vrp_data.customers[cust_index].y, individual.vrp_data.depots[last_cust].y], colors[index])
                else:
                    # plot arc between customers
                    plt.plot([individual.vrp_data.customers[cust_index].x, individual.vrp_data.customers[last_cust].x], [
                        individual.vrp_data.customers[cust_index].y, individual.vrp_data.customers[last_cust].y], colors[index])
                last_cust = cust_index
            # no customers in tour
            if last_cust < individual.vrp_data.nr_depots:
                plt.plot([individual.vrp_data.depots[depot_index].x, individual.vrp_data.depots[last_cust].x], [
                         individual.vrp_data.depots[depot_index].y, individual.vrp_data.depots[last_cust].y], colors[index])
            # customers in tour, from last customer to depot
            else:
                plt.plot([individual.vrp_data.depots[depot_index].x, individual.vrp_data.customers[last_cust].x], [
                    individual.vrp_data.depots[depot_index].y, individual.vrp_data.customers[last_cust].y], colors[index])
    plt.show()


if __name__ == "__main__":
    test()
