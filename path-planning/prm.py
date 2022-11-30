import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import collections as mc

def generate_obstacles(bottom_left, top_right, min_size, max_size, n=10):
    obstacles = []
    for _ in range(n):
        x = np.random.uniform(bottom_left[0], top_right[0])
        y = np.random.uniform(bottom_left[1], top_right[1])
        width = np.random.uniform(min_size[0], max_size[0])
        height = np.random.uniform(min_size[1], max_size[1])
        obstacles.append(((x, y), width, height))
    return obstacles

def inobstacle(v, obstacles, epsilon=0):
    for obs in obstacles:
        (x, y), width, height = obs
        if x - epsilon <= v[0] <= x + width + epsilon and y - epsilon <= v[1] <= y + height + epsilon:
            return True
    return False

def obstacle(v1, v2, obstacles, epsilon=0):
    for obs in obstacles:
        (x, y), width, height = obs
        m = (v2[1] - v1[1]) / (v2[0] - v1[0])
        b = v1[1] - m * v1[0]
        if x - epsilon <= (y - b) / m <= x + width + epsilon:
            return True
        elif x - epsilon <= (y + height - b) / m <= x + width + epsilon:
            return True
        elif y - epsilon <= m * x + b <= y + height + epsilon:
            return True
        elif y - epsilon <= m * (x + width) + b <= y + height + epsilon:
            return True
    return False

def random_position(bottom_left, top_right):
    x = np.random.uniform(bottom_left[0], top_right[0])
    y = np.random.uniform(bottom_left[1], top_right[1])
    return x, y

def nearest(vertex_set, vertex, k):
    v_nearest = []
    for v in vertex_set:
        dist = np.sqrt((v[0] - vertex[0]) ** 2 + (v[1] - vertex[1]) ** 2)
        v_nearest.append((v, dist))
    v_nearest = sorted(v_nearest, key=lambda x: x[1])
    v_nearest = [v for v, _ in v_nearest][1:k + 1]
    return v_nearest

def roadmap(obstacles, bottom_left, top_right, k=5, epsilon=0, n_iterations=100):
    vertex_set = set()
    edge_set = set()
    while len(vertex_set) < n_iterations:
        v = random_position(bottom_left, top_right)
        if not inobstacle(v, obstacles, epsilon=epsilon):
            vertex_set.add(v)
    for v in vertex_set:
        neighbors = nearest(vertex_set, v, k=k)
        for neighbor in neighbors:
            if (v, neighbor) in edge_set or (neighbor, v) in edge_set:
                continue
            if obstacle(v, neighbor, obstacles, epsilon=epsilon):
                continue
            edge_set.add((v, neighbor))
    return vertex_set, edge_set

def connect(vertex, vertex_set, obstacles, k=5, epsilon=0):
    v_nearest = []
    for v in vertex_set:
        dist = np.sqrt((v[0] - vertex[0]) ** 2 + (v[1] - vertex[1]) ** 2)
        v_nearest.append((v, dist))
    v_nearest = sorted(v_nearest, key=lambda x: x[1])
    v_nearest = [v for v, _ in v_nearest][:k]
    for v in v_nearest:
        if obstacle(v, vertex, obstacles, epsilon=epsilon):
            continue
        return (v, vertex)
    
def adjacency_list(edge_set):
    graph = dict()
    for (v1, v2) in edge_set:
        graph[v1] = graph[v1] + [v2] if v1 in graph else [v2]
        graph[v2] = graph[v2] + [v1] if v2 in graph else [v1]
    return graph

def path(node, memory):
    if not node:
        return set(), set()
    vertex_set, edge_set = path(memory[node][0], memory) 
    vertex_set |= {node}
    edge_set = edge_set | {(node, memory[node][0])} if memory[node][0] else edge_set
    return vertex_set, edge_set

def a_star(adj_list, start, goal):
    explored = set()
    queue = [(0, start)]
    memory = {start: (None, 0)}
    while queue:
        cost, node = heapq.heappop(queue)
        explored.add(node)
        if node == goal:
            return path(goal, memory)
        neighbors = adj_list[node]
        for neighbor in neighbors:
            if neighbor in explored:
                continue
            neighbor_cost = np.sqrt((neighbor[0] - node[0]) ** 2 + (neighbor[1] - node[1]) ** 2)
            future_cost = np.sqrt((neighbor[0] - goal[0]) ** 2 + (neighbor[1] - goal[1]) ** 2)
            total_cost = cost + neighbor_cost + future_cost
            if neighbor not in memory or total_cost < memory[neighbor][1]:
                heapq.heappush(queue, (total_cost, neighbor))
                memory[neighbor] = (node, total_cost)

def prm(graph, obstacles, start, goal):
    vertex_set, edge_set = graph
    start_edge = connect(start, vertex_set, obstacles)
    if not start_edge:
        return
    goal_edge = connect(goal, vertex_set, obstacles)
    if not goal_edge:
        return
    vertex_set |= {start, goal}
    edge_set |= {start_edge, goal_edge}
    adj_list = adjacency_list(edge_set)
    path_graph = a_star(adj_list, start, goal)
    if not path_graph:
        return
    graph = vertex_set, edge_set
    return graph, path_graph

obstacles = generate_obstacles(
    bottom_left=(50, 50), 
    top_right=(400, 400), 
    min_size=(25, 25), 
    max_size=(50, 50)
)

graph = roadmap(obstacles, bottom_left=(0, 0), top_right=(500, 500), n_iterations=1000)
result = prm(graph, obstacles, start=(20, 20), goal=(475, 475))
if not result:
    exit('prm algorithm failed')
graph, path_graph = result
    
fig, ax = plt.subplots()
for obs in obstacles:
    ax.add_patch(Rectangle(*obs))
lc1 = mc.LineCollection(graph[1])
ax.add_collection(lc1)
ax.scatter([v[0] for v in graph[0]], [v[1] for v in graph[0]])
lc2 = mc.LineCollection(path_graph[1], color='r')
ax.add_collection(lc2)
ax.scatter([v[0] for v in path_graph[0]], [v[1] for v in path_graph[0]], color='r')

ax.set_xlim(0, 500)
ax.set_ylim(0, 500)

plt.show()