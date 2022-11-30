graph = {
  'A': ['B', 'C', 'D'], 
  'B': ['A', 'D', 'E'], 
  'C': ['A', 'D'], 
  'D': ['B', 'C', 'A', 'E'], 
  'E': ['B', 'D']
}

def construct_path(node, memory):
    if not node:
        return []
    return construct_path(memory[node], memory) + [node]

def bfs(graph, source, target):
    explored = set()
    queue = [source]
    memory = {source: None}
    while queue:
        node = queue.pop(0)
        explored.add(node)
        if node == target:
            return construct_path(target, memory)
        neighbors = graph[node]
        for neighbor in neighbors:
            if neighbor in explored or neighbor in queue:
                continue
            memory[neighbor] = node
            queue.append(neighbor)

print(bfs(graph, 'A', 'C'))




