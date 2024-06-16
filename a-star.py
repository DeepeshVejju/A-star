import sys
import math
from PIL import Image
import numpy as np
import pandas as pd
import heapq

"""
Class Node to hold all required values of a node in terrain
"""


class Node:

    def __init__(self, x, y, g, f, parent):
        self.x = x
        self.y = y
        self.g = g
        self.f = f
        self.parent = parent

    def __members(self):
        return (self.x, self.y)

    def __eq__(self, point):
        return self.__members() == point.__members()

    def __hash__(self):
        return hash(self.__members())

    def __lt__(self, other):
        return self.f < other.f


"""
Speed Dict to store values of speed factors over different terrains.
"""
speed_dict = dict()
speed_dict[str(np.array([0, 0, 255]))] = 10  # Lake/Swamp/Marsh
speed_dict[str(np.array([5, 73, 24]))] = 3  # Impassible vegetation
speed_dict[str(np.array([255, 192, 0]))] = 2  # Rough meadow
speed_dict[str(np.array([2, 136, 40]))] = 1.5  # Walk forest
speed_dict[str(np.array([2, 208, 60]))] = 1.25  # Slow run forest
speed_dict[str(np.array([255, 255, 255]))] = 1  # Easy movement forest
speed_dict[str(np.array([248, 148, 18]))] = 0.9  # Open land
speed_dict[str(np.array([0, 0, 0]))] = 0.5  # Footpath
speed_dict[str(np.array([71, 51, 3]))] = 0.2  # Paved road


def heuristic(x1, x2, y1, y2, e1, e2):
    """
    Euclidean heuristic to calculate distance.
    """
    d = math.sqrt(((x2 - x1) * 7.55) ** 2 + ((y2 - y1) * 10.29) ** 2 + ((e2 - e1) ** 2))
    return d


def get_neighbors(x, y, pixels):
    """
    Method to return neighbors of a node.
    """
    list_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for val in list_neighbors:
        a, b = val
        if (0 <= x + a <= 499) and (0 <= y + b <= 394):
            if np.array_equal(pixels[x + a][y + b], [205, 0, 101]):  # Out of Bounds
                pass
            else:
                neighbors.append((x + a, y + b))

    return neighbors


def a_star(pixels, elevation, queue, visited, src, dest):
    """
    A star to calculate path from source to destination.
    """
    queue_set = set()
    heapq.heappush(queue, src)
    queue_set.add(src)
    while len(queue) > 0:

        node = heapq.heappop(queue)
        neighbors = get_neighbors(node.x, node.y, pixels)

        if node == dest:
            path = []
            while node:
                path.append((node.x, node.y))
                node = node.parent
            return path
        for val in neighbors:
            a, b = val
            if val in visited:
                continue
            neighbor_g = node.g + heuristic(node.x, a, node.y, b, elevation.iloc[node.x, node.y],
                                            elevation.iloc[a, b])
            neighbor_h = heuristic(dest.x, a, dest.y, b, elevation.iloc[dest.x, dest.y], elevation.iloc[a, b])
            neighbor_f = neighbor_g + neighbor_h
            neighbor_f *= speed_dict[str(pixels[a][b])]
            neighbor = Node(a, b, neighbor_g, neighbor_f, node)

            if neighbor not in queue_set:
                heapq.heappush(queue, neighbor)
                queue_set.add(neighbor)
            elif neighbor_f < neighbor.f:
                neighbor.f = neighbor_f
                neighbor.parent = node
        visited.add((node.x, node.y))


def get_pixels(filename):
    """
    Method to return rgb values of pixels.
    """
    terrain = Image.open(filename)
    terrain_rgb = terrain.convert('RGB')
    terrain_rgb_array = np.array(terrain_rgb)
    return terrain_rgb_array


def get_elevations(filename):
    """
    Method to extract elevations of terrain.
    """
    df = pd.read_csv(filename, delim_whitespace=True, header=None)
    df = df.iloc[:, :-5]
    return df


def get_path(filename):
    """
    Method to extract sources and destination coordinates.
    """
    df = pd.read_csv(filename, delim_whitespace=True, header=None)
    return df


def image_path(path_pixels, pixels, filename):
    """
    Method to draw path from source to destination on image.
    """
    for path in path_pixels:
        for i in range(0, len(path)):
            a, b = path[i]
            pixels[a][b] = np.array([200, 100, 230])
    new_image = Image.fromarray(pixels)
    new_image.save(filename)


def read_input():
    """
    Method to read input files and perform astar.
    """

    try:
        # this checks if correct number of arguments are given.
        if len(sys.argv) != 5:
            print("Usage: python3 lab1.py terrain.png mpp.txt red.txt redOut.png")
            exit(1)
        else:

            pixels = get_pixels(sys.argv[1])
            elevations = get_elevations(sys.argv[2])
            path = get_path(sys.argv[3])
            output_file = sys.argv[4]

            path_np = np.array(path)
            distance = 0
            path_pixels = []

            for i in range(0, len(path_np) - 1):
                a = path_np[i][1]
                b = path_np[i][0]
                c = path_np[i + 1][1]
                d = path_np[i + 1][0]

                queue = []
                visited = set()
                src = Node(a, b, 0, 0, None)
                dest = Node(c, d, 0, 0, None)
                path = a_star(pixels, elevations, queue, visited, src, dest)
                if path is None:
                    print("No route from source to destination.")
                path_pixels.append(path)
                for i in range(0, len(path) - 1):
                    a, b = path[i]
                    c, d = path[i + 1]
                    distance += heuristic(a, c, b, d, elevations.iloc[a, b], elevations.iloc[c, d])

            print(distance)
            image_path(path_pixels, pixels, output_file)


    except FileNotFoundError:
        print("File not found:", sys.argv[1])
        exit(1)


if __name__ == '__main__':
    """
    Main method to start program.
    """
    read_input()
