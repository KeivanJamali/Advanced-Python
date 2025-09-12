import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Players import Intersection
import sympy as sp 


class Graph_Generator:
    def __init__(self, network_files: list, 
                 dedicated_lane_length: int, 
                 lane_changing_zone_length: int,
                 each_block_length: int):
        """Initialize the traffic network graph.
        
        Args:
            network_files (list): List containing:
                [0]: Path to CSV file with network topology
                [1]: Path to text file with node positions
            dedicated_lane_length (int): Length of AV-only lanes in meters
            lane_changing_zone_length (int): Length of lane changing zones in meters
            each_block_length (int): Length of each road block in meters
        """
        self.network_data = pd.read_csv(network_files[0])
        self.pos = pd.read_csv(network_files[1], sep=r"\s+",)
        self.graph = nx.DiGraph()
        
        # Create graph edges with travel time functions
        for _, row in self.network_data.iterrows():
            x, expr = self._generate_travel_time(length=int(row["length"]))
            self.graph.add_edge(str(row["from"]), str(row["to"]), length=row["length"], param=x, expr=expr)

        for node in self.graph.nodes:
            neighbors = list(self.graph.predecessors(node))
            lengths = [self.graph[nbr][node].get("length", 1) for nbr in neighbors]
            self.graph.nodes[node]["intersection"] = Intersection(node_id=str(node),
                                                                  neighbors=neighbors,
                                                                  lengths=lengths,
                                                                  dedicated_lane_length=dedicated_lane_length,
                                                                  lane_changing_zone_length=lane_changing_zone_length,
                                                                  each_block_length=each_block_length)

    def _generate_travel_time(self, length: int) -> tuple[sp.Symbol, sp.Expr]:
        """Generate a BPR (Bureau of Public Roads) travel time function for a road segment.
        
        This private method creates a symbolic expression for travel time based on the
        BPR formula: t = t0 * (1 + α * (x/c)^β)
        where:
        - t0 = free flow travel time (length/speed)
        - x = traffic flow (variable)
        - c = road capacity
        - α, β = calibration parameters
        
        Args:
            length (int): Length of the road segment in meters
            
        Returns:
            tuple: (x, expr) where:
                - x: Sympy symbol representing traffic flow
                - expr: Sympy expression for travel time function
                
        Note:
            Default parameters:
            - Base speed: 60 m/s
            - Capacity: length/5 vehicles
            - α (alpha): 0.15
            - β (beta): 4
        """
        x = sp.symbols("x")
        capacity = length / 5
        speed = 60
        alpha = 0.15
        beta = 4
        expr = (length / speed) * (1 + alpha * (x / capacity)**beta)
        return x, expr

    def draw(self) -> None:
        """Visualize the road network using matplotlib.
        
        This method draws:
        - Nodes (intersections) as light blue circles
        - Directed edges (road segments) as arrows
        - Edge labels showing road segment lengths
        - Node labels showing intersection IDs
        
        Uses the actual geographical coordinates from the position file
        for node placement.
        
        Example:
            >>> network = Graph_Generator(...)
            >>> network.draw()
            # Displays the network visualization
        """
        # Create position dictionary from the pos DataFrame
        pos = {str(row['Node']): (row['X'], row['Y']) for _, row in self.pos.iterrows()}
        
        # Draw the network
        nx.draw(self.graph, pos, with_labels=True, 
                node_size=1500, node_color="lightblue", arrowsize=20)

        # Add edge labels showing lengths
        edge_labels = nx.get_edge_attributes(self.graph, 'length')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.show()