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


def path_cost(x, y, show=False, neighbor_size=None):
    """
    type of x & y: np.array
    type of neighbor_size: int
    """
    x_origin = copy.deepcopy(x)
    y_origin = copy.deepcopy(y)

    # --- distance & accumulated distance --- #
    distances = np.zeros((len(y), len(x)))
    accumulated_cost = np.zeros((len(y), len(x)))

    # calculate distance
    for i in range(len(y)):
        for j in range(len(x)):
            distances[i, j] = (x[j] - y[i]) ** 2

    # calculate distance (accumulated)
    if neighbor_size is None:
        for i in range(1, len(y)):
            for j in range(1, len(x)):
                accumulated_cost[i, j] = min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                             accumulated_cost[i, j - 1]) + distances[i, j]
    elif isinstance(neighbor_size, int):
        # todo: address neighbor size to reduce computation
        for i in range(1, len(y)):
            for j in range(1, len(x)):
                accumulated_cost[i, j] = min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                             accumulated_cost[i, j - 1]) + distances[i, j]

    # optimal path & cost
    path = [[len(x) - 1, len(y) - 1]]
    cost = 0
    i = len(y) - 1
    j = len(x) - 1
    while i > 0 and j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if accumulated_cost[i - 1, j] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                 accumulated_cost[i, j - 1]):
                i = i - 1
            elif accumulated_cost[i, j - 1] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                   accumulated_cost[i, j - 1]):
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append([j, i])
    path.append([0, 0])
    for [y, x] in path:
        cost = cost + distances[x, y]

    # plot
    if show:
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
        plt.show(block=True)

    return path, cost


# example
x = np.array([1, 1, 2, 3, 2, 0])
y = np.array([0, 1, 1, 2, 3, 2, 1])

path, cost = path_cost(x, y, show=True)
