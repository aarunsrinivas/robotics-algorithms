import heapq

graph = {
    'A': [('B', 9), ('C', 4), ('D', 7)],
    'B': [('A', 9), ('E', 11)],
    'C': [('A', 4), ('E', 17), ('F', 12)],
    'D': [('A', 7), ('F', 14)],
    'E': [('B', 11), ('C', 17), ('Z', 5)],
    'F': [('C', 12), ('D', 14), ('Z', 9)],
    'Z': [('E', 5), ('F', 9)]
}

heuristic = {'A': 21, 'B': 14, 'C': 18, 'D': 18, 'E': 5, 'F': 8, 'Z': 0}

def construct_path(node, memory):
    if not node:
        return []
    return construct_path(memory[node][0], memory) + [node]

def a_star(graph, heuristic, source, target):
    explored = set()
    queue = [(0, source)]
    memory = {source: (None, 0)}
    while queue:
        cost, node = heapq.heappop(queue)
        explored.add(node)
        if node == target:
            return construct_path(target, memory)
        neighbors = graph[node]
        for neighbor, neighor_cost in neighbors:
            if neighbor in explored:
                continue
            future_cost = heuristic[neighbor]
            total_cost = cost + neighor_cost + future_cost
            if neighbor not in memory or total_cost < memory[neighbor][1]:
                heapq.heappush(queue, (total_cost, neighbor))
                memory[neighbor] = (node, total_cost)

print(a_star(graph, heuristic, 'A', 'Z'))