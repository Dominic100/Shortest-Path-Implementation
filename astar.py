import heapq
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import time
import math

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

def heuristic(node, goal, pos):
    # Simple Euclidean distance heuristic
    x1, y1 = pos[node]
    x2, y2 = pos[goal]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def a_star(graph, start, goal, pos):
    # Initialization
    open_set = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal, pos)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, weight in graph[current].items():
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal, pos)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Adjust the figsize to make more room for the table
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

        # Add columns for heuristic, sum, and chosen path weight
        table_data_with_columns = []
        for data in table_data:
            source, connections, weights, destination, weight = data
            heuristic_values = [heuristic(source, neighbor, pos) for neighbor in graph[source].keys()]
            sum_value = ", ".join(f"{weight + heuristic_value:.4f}" for heuristic_value in heuristic_values)
            chosen_path_weight = min(graph[source].values())
            table_data_with_columns.append((source, connections, sum_value, f"{chosen_path_weight:.4f}"))  # Limit the decimal places

        ax2.axis('off')
        ax2.table(cellText=table_data_with_columns, colLabels=['From', 'Possible Connections', 'Sum', 'Chosen Path Weight'], loc='center')

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
    
    path = a_star(graph, source, destination, pos)
    
    if path:
        print("Shortest Path:", path)
        print("Shortest Distance:", sum(graph[path[i]][path[i+1]] for i in range(len(path)-1)))

        # Simulate routing
        simulate_routing(graph, path, pos)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()


