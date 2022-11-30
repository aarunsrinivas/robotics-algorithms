import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import collections as mc

def generate_obstacles(min_dim, max_dim, min_size, max_size, n=10):
    obstacles = []
    for _ in range(n):
        x = np.random.uniform(min_dim[0], max_dim[0])
        y = np.random.uniform(min_dim[1], max_dim[1])
        width = np.random.uniform(min_size[0], max_size[0])
        height = np.random.uniform(min_size[1], max_size[1])
        obstacles.append(((x, y), width, height))
    return obstacles

def intersect(v1, v2, obstacle, e=0):
    (x, y), width, height = obstacle
    m = (v2[1] - v1[1]) / (v2[0] - v1[0])
    b = v1[1] - m * v1[0]
    if x - e <= (y - b) / m <= x + width + e:
        return True
    elif x - e <= (y + height - b) / m <= x + width + e:
        return True
    elif y - e <= m * x + b <= y + height + e:
        return True
    elif y - e <= m * (x + width) + b <= y + height + e:
        return True
    return False

def find_closest_vertex(v, graph):
    min_dist = 1e9
    closest_vertex = None
    for vertex in graph:
        dist = np.sqrt((v[0] - vertex[0]) ** 2 + (v[1] - vertex[1]) ** 2)
        if dist < min_dist:
            closest_vertex = vertex
            min_dist = dist
    return closest_vertex


def rrt(start, goal, obstacles, min_dim, max_dim, e=50, n_iterations=1000):
    graph = {start: []}
    for _ in range(n_iterations):
        x = np.random.uniform(min_dim[0], max_dim[0])
        y = np.random.uniform(min_dim[1], max_dim[1])
        closest_vertex = find_closest_vertex((x, y), graph)
        flag = False
        for obstacle in obstacles:
            if intersect((x, y), closest_vertex, obstacle):
                flag = True
                break
        if flag:
            continue
        graph[closest_vertex].append((x, y))
        graph[(x, y)] = [closest_vertex]
        g_dist = np.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)
        if g_dist < e:
            return graph
    return graph

def construct_path(node, memory):
    if not node:
        return []
    return construct_path(memory[node][0], memory) + [node]

def a_star(graph, start, goal, e=10):
    explored = set()
    queue = [(0, start)]
    memory = {start: (None, 0)}
    while queue:
        cost, node = heapq.heappop(queue)
        explored.add(node)
        g_dist = np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)
        if g_dist < e:
            return construct_path(node, memory)
        neighbors = graph[node]
        for neighbor in neighbors:
            if neighbor in explored:
                continue
            neighbor_cost = np.sqrt((neighbor[0] - node[0]) ** 2 + (neighbor[1] - node[1]) ** 2)
            future_cost = np.sqrt((neighbor[0] - goal[0]) ** 2 + (neighbor[1] - goal[1]) ** 2)
            total_cost = cost + neighbor_cost + future_cost
            if neighbor not in memory or total_cost < memory[neighbor][1]:
                heapq.heappush(queue, (total_cost, neighbor))
                memory[neighbor] = (node, total_cost)


start = (20, 20)
goal = (475, 475)
obstacles = generate_obstacles((50, 50), (400, 400), (25, 25), (50, 50))

graph = rrt(start, goal, obstacles, (0, 0), (500, 500))
lines = []
for k in graph:
    for v in graph[k]:
        lines.append([k, v])
points = a_star(graph, start, goal, e=50)
path = []
for i in range(len(points) - 1):
    path.append([points[i], points[i + 1]])

fig, ax = plt.subplots()
lc1 = mc.LineCollection(lines)
ax.add_collection(lc1)
for obstacle in obstacles:
    ax.add_patch(Rectangle(*obstacle))
ax.scatter([k[0] for k in graph], [k[1] for k in graph])

lc2 = mc.LineCollection(path, color='r')
ax.add_collection(lc2)
ax.scatter([p[0] for p in points], [p[1] for p in points], color='r')

ax.set_xlim(0, 500)
ax.set_ylim(0, 500)

plt.show()