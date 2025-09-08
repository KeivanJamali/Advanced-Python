import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class Graph_Generator:
    def __init__(self, network_file: str):
        self.network_data = pd.read_csv(network_file)
        self.graph = nx.DiGraph()
        for _, row in self.network_data.iterrows():
            self.graph.add_edge(row["from"], row["to"], length=row["length"])

    def draw(self):
        pos = nx.spring_layout(self.graph, seed=42)  # I want to have the actuall coordinate of the soufalls network here.
        nx.draw(self.graph, pos, with_labels=True, node_size=1500, node_color="lightblue", arrowsize=20)

        edge_labels = nx.get_edge_attributes(self.graph, 'length')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.show()