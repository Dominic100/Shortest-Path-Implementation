import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

# Define the graph
graph = {
    'A': {'B': 1, 'C': 4, 'D': 2},
    'B': {'A': 1, 'C': 2, 'D': 5, 'E': 3},
    'C': {'A': 4, 'B': 2, 'D': 1, 'F': 4},
    'D': {'A': 2, 'B': 5, 'C': 1, 'E': 3, 'F': 6},
    'E': {'B': 3, 'D': 3, 'F': 2},
    'F': {'C': 4, 'D': 6, 'E': 2, 'G': 3},
    'G': {'F': 3}
}

def bellman_ford(graph, start):
    # Initialization
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {}

    # Iterate |V| - 1 times
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    previous[neighbor] = node

    # Check for negative cycles
    for node in graph:
        for neighbor, weight in graph[node].items():
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative cycle")

    return distances, previous

def plot_graph(graph):
    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500)

    plt.title("Bellman-Ford Algorithm")
    plt.grid(False)

    return pos

def animate_bellman_ford(graph, start, destination, pos):
    distances, previous = bellman_ford(graph, start)

    if destination not in previous:
        print("Destination node is not reachable from the source node.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)

def animate_bellman_ford(graph, start, destination, pos):
    distances, previous = bellman_ford(graph, start)

    if destination not in previous:
        print("Destination node is not reachable from the source node.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)

    def update(frame, distances, previous):
        ax1.clear()
        ax2.clear()
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, ax=ax1)
        ax1.set_title("Bellman-Ford Algorithm (Step {})".format(frame + 1))
        ax1.grid(False)

        # Perform Bellman-Ford relaxation for one step
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    previous[neighbor] = node

        # Highlight single path
        path = []
        node = destination
        while node != start and node is not None:
            path.append(node)
            node = previous.get(node)
        path.reverse()
        if path:
            current_node = path[min(frame, len(path) - 1)]
            nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_color='red', node_size=1500, ax=ax1)
            if frame > 0:
                previous_node = path[min(frame - 1, len(path) - 1)]
                nx.draw_networkx_edges(G, pos, edgelist=[(previous_node, current_node)], width=2, edge_color='red', ax=ax1)

        # Update table data with the current distances
        table_data = []
        for node, distance in distances.items():
            if distance == float('inf'):
                status = 'Unreachable'
            else:
                status = 'Optimal'
            table_data.append((node, distance, status))

        ax2.axis('off')
        ax2.table(cellText=table_data, colLabels=['Node', 'Distance', 'Status'], loc='center')

        # Stop the animation if the destination node is reached
        if current_node == start:
            ani.event_source.stop()

    ani = animation.FuncAnimation(fig, update, frames=len(graph), fargs=(distances, previous), interval=1000, repeat=False)
    plt.show()



def get_node_clicked(pos):
    print("Click on the source node.")
    source_node = plt.ginput(n=1, timeout=-1)[0]
    print("Selected source node:", source_node)

    plt.text(0.5, 1.1, 'Click on the destination node', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, 1.05, '(Close the window to cancel)', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    print("Click on the destination node.")
    destination_node = plt.ginput(n=1, timeout=-1)[0]
    print("Selected destination node:", destination_node)

    return source_node, destination_node

def find_nearest_node(coords, pos):
    min_dist = float('inf')
    nearest_node = None
    for node, node_coords in pos.items():
        dist = (coords[0] - node_coords[0])**2 + (coords[1] - node_coords[1])**2
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

def main():
    pos = plot_graph(graph)
    source_node, destination_node = get_node_clicked(pos)
    
    # Convert clicked coordinates to node labels
    start = find_nearest_node(source_node, pos)
    destination = find_nearest_node(destination_node, pos)
    
    if start is None or destination is None or start not in graph or destination not in graph:
        print("Invalid source or destination node.")
        return
    
    animate_bellman_ford(graph, start, destination, pos)

if __name__ == "__main__":
    main()
