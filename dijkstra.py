import heapq
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import time

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

def dijkstra(graph, start):
    # Initialization
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    previous = {}

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        # Skip if we already found a better way
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # Update distance if shorter path found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(heap, (distance, neighbor))

    return distances, previous

def shortest_path(previous, start, end):
    path = []
    while end:
        path.append(end)
        end = previous.get(end)
    return path[::-1]

def plot_graph(graph):
    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500)

    plt.title("Select Source and Destination Nodes")
    plt.grid(False)

    return pos

def simulate_routing(graph, path, pos):
    routing_info = []
    for i in range(len(path) - 1):
        source = path[i]
        destination = path[i + 1]
        routing_info.append((source, ', '.join(graph[source].keys()), ', '.join(str(weight) for weight in graph[source].values()), destination, graph[source][destination]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)

    def update(frame):
        ax1.clear()
        ax2.clear()
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, ax=ax1)
        ax1.set_title("Packet Routing (Time step: {})".format(frame))
        ax1.grid(False)

        # Highlight only the current segment of the path
        for i in range(min(frame + 1, len(path) - 1)):
            source = path[i]
            destination = path[i + 1]
            nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='red', node_size=1500, ax=ax1)
            nx.draw_networkx_nodes(G, pos, nodelist=[destination], node_color='green', node_size=1500, ax=ax1)
            nx.draw_networkx_edges(G, pos, edgelist=[(source, destination)], width=2, edge_color='red', ax=ax1)

        table_data = routing_info[:frame+1]
        ax2.axis('off')
        ax2.table(cellText=table_data, colLabels=['From', 'Possible Connections', 'Weights', 'Chosen Path', 'Weight (Chosen Path)'], loc='center')

    ani = animation.FuncAnimation(fig, update, frames=len(path)-1, interval=1000, repeat=False)
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
    source = find_nearest_node(source_node, pos)
    destination = find_nearest_node(destination_node, pos)
    
    if source is None or destination is None or source not in graph or destination not in graph:
        print("Invalid source or destination node.")
        return
    
    distances, previous = dijkstra(graph, source)
    path = shortest_path(previous, source, destination)
    
    print("Shortest Path:", path)
    print("Shortest Distance:", distances[destination])

    # Simulate routing
    simulate_routing(graph, path, pos)

if __name__ == "__main__":
    main()
