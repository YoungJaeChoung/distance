# https://nipunbatra.github.io/blog/2014/dtw.html
import numpy as np
import matplotlib.pyplot as plt
import copy


def distance_cost_plot(distances, path_x=None, path_y=None):
    plt.imshow(distances, interpolation='nearest', cmap='Reds')
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.colorbar()
    if (path_x is not None) and (path_y is not None):
        plt.plot(path_x, path_y)
    plt.show()


def path_cost(x, y, plot=False, name=None, neighbor_size=None):
    """ calculate dynamic time warping distance

    Parameters
    ----------
    x, y: (np.array)
    plot: (bool) plot or not
    name: (str) name of plot for saving file
    neighbor_size: (int) the size of neighborhoods included in computation`
    """
    if (plot is True) & (name is None):
        raise NotImplementedError("Plot is True, but name is None")

    x_origin = copy.deepcopy(x)
    y_origin = copy.deepcopy(y)

    # --- distance & accumulated distance --- #
    distances = np.zeros((len(y), len(x)))
    accumulated_cost = np.zeros((len(y), len(x)))

    # calculate distance
    for idx_y in range(len(y)):
        for idx_x in range(len(x)):
            distances[idx_y, idx_x] = (x[idx_x] - y[idx_y]) ** 2

    # calculate distance (accumulated)
    if neighbor_size is None:
        for idx_y in range(0, len(y)):
            for idx_x in range(0, len(x)):
                """ check 
                i, j = 1, 0
                """
                """ original simple format
                for idx_y in range(1, len(y)):
                    for idx_x in range(1, len(x)):
                        accumulated_cost[idx_y, idx_x] = min(accumulated_cost[idx_y - 1, idx_x - 1],
                                             accumulated_cost[idx_y - 1, idx_x],
                                             accumulated_cost[idx_y, idx_x - 1]) \
                                         + distances[idx_y, idx_x]
                """
                if (idx_y == 0) and (idx_x == 0):
                    accumulated_cost[idx_y, idx_x] = distances[idx_y, idx_x]
                elif (idx_y == 0) and (idx_x >= 1):
                    accumulated_cost[idx_y, idx_x] = accumulated_cost[idx_y, idx_x - 1] \
                                                     + distances[idx_y, idx_x]
                elif (idx_y >= 1) and (idx_x == 0):
                    accumulated_cost[idx_y, idx_x] = accumulated_cost[idx_y - 1, idx_x] \
                                                     + distances[idx_y, idx_x]
                else:
                    accumulated_cost[idx_y, idx_x] = min(accumulated_cost[idx_y - 1, idx_x - 1],
                                                         accumulated_cost[idx_y - 1, idx_x],
                                                         accumulated_cost[idx_y, idx_x - 1]) \
                                                     + distances[idx_y, idx_x]
    elif isinstance(neighbor_size, int):
        large_num = np.sum(distances)
        for idx_y in range(0, len(y)):
            for idx_x in range(0, len(x)):
                if abs(idx_x - idx_y) <= neighbor_size:
                    if (idx_y == 0) and (idx_x == 0):
                        accumulated_cost[idx_y, idx_x] = distances[idx_y, idx_x]
                    elif (idx_y == 0) and (idx_x >= 1):
                        accumulated_cost[idx_y, idx_x] = accumulated_cost[idx_y, idx_x - 1] \
                                                         + distances[idx_y, idx_x]
                    elif (idx_y >= 1) and (idx_x == 0):
                        accumulated_cost[idx_y, idx_x] = accumulated_cost[idx_y - 1, idx_x] \
                                                         + distances[idx_y, idx_x]
                    else:
                        accumulated_cost[idx_y, idx_x] = min(accumulated_cost[idx_y - 1, idx_x - 1],
                                                             accumulated_cost[idx_y - 1, idx_x],
                                                             accumulated_cost[idx_y, idx_x - 1]) \
                                                         + distances[idx_y, idx_x]
                else:
                    accumulated_cost[idx_y, idx_x] = large_num

    # optimal path & cost
    path = [[len(x) - 1, len(y) - 1]]
    cost = 0
    idx_y = len(y) - 1
    idx_x = len(x) - 1
    while idx_y > 0 and idx_x > 0:
        if idx_y == 0:
            idx_x = idx_x - 1
        elif idx_x == 0:
            idx_y = idx_y - 1
        else:
            if accumulated_cost[idx_y - 1, idx_x] == min(accumulated_cost[idx_y - 1, idx_x - 1],
                                                         accumulated_cost[idx_y - 1, idx_x],
                                                         accumulated_cost[idx_y, idx_x - 1]):
                idx_y = idx_y - 1
            elif accumulated_cost[idx_y, idx_x - 1] == min(accumulated_cost[idx_y - 1, idx_x - 1],
                                                           accumulated_cost[idx_y - 1, idx_x],
                                                           accumulated_cost[idx_y, idx_x - 1]):
                idx_x = idx_x - 1
            else:
                idx_y = idx_y - 1
                idx_x = idx_x - 1
        path.append([idx_x, idx_y])
    path.append([0, 0])
    for [y, x] in path:
        cost = cost + distances[x, y]

    #
    # plot
    #
    if plot:
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.plot(x_origin, 'bo-', label='x')
        plt.plot(y_origin, 'g^-', label='y')
        plt.legend()
        for [map_x, map_y] in path:
            plt.plot([map_x, map_y], [x_origin[map_x], y_origin[map_y]], 'r')

        f.add_subplot(1, 2, 2)
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        distance_cost_plot(accumulated_cost, path_x, path_y)
        # plt.show(block=True)
        plt.savefig(name)
        plt.close()

    return path, cost


# example
x = np.array([1, 1, 2, 3, 2, 0])
y = np.array([0, 1, 1, 2, 3, 2, 1])

path, cost = path_cost(x, y, plot=True, name="DTW", neighbor_size=None)